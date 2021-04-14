#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (author: Liyong Guo)
# Apache 2.0

import logging
import os
import yaml
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

from model import TransformerModel

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import decoders
from common import load_checkpoint
from model import TransformerModel
from typing import Dict, List

import k2


def word_seqs_to_list_str(word_seqs: k2.RaggedInt,
                          symbol_table: k2.SymbolTable) -> List[str]:
    '''
    Args:
      word_seqs:[path][word]
    '''
    word_ids = word_seqs.values()
    words = [symbol_table.get(word_idx.item()) for word_idx in word_ids]
    ragged_shape = word_seqs.row_splits(1)
    sentences = []
    for idx, start_idx in enumerate(ragged_shape[:-1]):
        sentences.append(' '.join(words[start_idx:ragged_shape[idx + 1]]))
    return sentences


def validate_configs(configs: Dict, required_fields: List) -> bool:
    not_exist_fields = []
    for field in required_fields:
        if field not in configs or configs[field] is None:
            not_exist_fields.append(field)
    if len(not_exist_fields) > 0:
        assert False, 'set following required fields {}'.format(
            ' '.join(not_exist_fields))
    return True


def extract_configs(config_file) -> Dict:
    assert os.path.exists(config_file), '{} does not exist'.format(cofnig_file)
    required_fields = [
        'model_module',
        'shared_conf',
    ]
    with open(config_file, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    validate_configs(configs, required_fields)

    model_conf = '{}_conf'.format(configs['model_module'])
    ntoken = configs['shared_conf']['ntoken']

    assert 'model_dir' in configs['trainer_conf']
    configs[model_conf]['ntoken'] = ntoken

    return configs


class Evaluator(object):

    def __init__(self,
                 device,
                 model_path,
                 config_file=None,
                 tokenizer_path=None,
                 words_txt=None,
                 batch_size=1):
        self.device = device
        configs = extract_configs(config_file)
        if configs['model_module'] == 'transformer':
            model = TransformerModel(**configs['transformer_conf'])
            if model_path is not None:
                assert os.path.exists(model_path)
                load_checkpoint(model_path, model)
        self.model = model
        self.ntoken = model.ntoken
        self.batch_size = batch_size
        self.word_count = 0
        self.token_count = 0
        self.total_examples = 0
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.decoder = decoders.WordPiece()
        self.bos_id = self.ntoken - 3
        self.eos_id = self.ntoken - 2
        self.pad_index = self.ntoken - 1
        if words_txt is not None:
            self.symbol_table = k2.SymbolTable.from_file(words_txt)

        self.criterion = nn.NLLLoss(ignore_index=self.pad_index,
                                    reduction='mean')

    def set_criterion(self, doing_rescore: bool):
        if doing_rescore:
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index,
                                        reduction='sum')
        else:
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index,
                                        reduction='mean')
    def reset_count_variables(self):
        self.word_count = 0
        self.token_count = 0
        self.total_examples = 0

    def batchify(self, txt_f):
        batch = []

        for line in txt_f:
            self.total_examples += 1
            line = line.strip().lower()

            token_id = self.tokenizer.encode(line).ids
            # +1 for <eos>
            self.word_count += len(line.split()) + 1
            # +1 for <eos>
            self.token_count += len(token_id) + 1
            token_id.insert(0, self.bos_id)
            token_id.append(self.eos_id)
            batch.append(token_id)
            if len(batch) == self.batch_size:
                # data_pad: [batch_size, seq_len]
                # each seq_len always different
                data_pad = pad_sequence(
                    [torch.from_numpy(np.array(x)).long() for x in batch],
                    True, self.pad_index)
                data_pad = data_pad.t().contiguous()
                # xs_pad, ys_pad: [max_seq_len, batch_size]
                # max_seq_len is the maximum length in current batch
                xs_pad = data_pad[:-1, :]
                ys_pad = data_pad[1:, :]
                yield xs_pad, ys_pad
                batch = []

    @torch.no_grad()
    def compute_ppl(self, txt_file: str):
        self.set_criterion(doing_rescore=False)
        # total_loss = torch.tensor([0.0]).to(self.device)
        # total_examples = torch.tensor([0.0]).to(self.device)
        # for batch_idx, batch in enumerate(self.dev_data_loader):
        total_loss = 0.0
        txt_f = open(txt_file, 'r')
        for batch_input, batch_target in self.batchify(txt_f):
            # batch_input: [seq_len, batch_size]
            # with contents: <bos> token_id token_id ....
            #
            # batch_target: [seq_len, batch_size]
            # with contensts: token_id token_id ... <eos>
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            batch_output = self.model(batch_input)

            prediction = batch_output.view(-1, self.ntoken)
            # target: [max_seq_len * batch_size]
            # example_1_token_1 example_2_token_1 example_3_token_1 .....
            target = batch_target.view(-1)
            loss = self.criterion(prediction, target)
            total_loss += loss * batch_input.shape[0]

        loss = total_loss / self.token_count
        token_ppl = math.exp(total_loss / self.token_count)
        word_ppl = math.exp(total_loss / self.word_count)
        log_str = 'dev examples: {} dev loss is {:.6f} and token_ppl {:.6f}  word_ppl {}'.format(
            int(self.total_examples), loss.item(), token_ppl, word_ppl)
        logging.info(log_str)
        txt_f.close()
        self.reset_count_variables()

    def batchify_sentences(self, sentences: List[str]):
        batch = []
        for line in sentences:
            self.total_examples += 1
            token_id = self.tokenizer.encode(line).ids
            # print('token_id: ', token_id)
            # +1 for <eos>
            self.word_count += len(line.split()) + 1
            # +1 for <eos>
            self.token_count += len(token_id) + 1

            token_id.insert(0, self.bos_id)
            token_id.append(self.eos_id)
            batch.append(token_id)
            if len(batch) == self.batch_size:
                # data_pad: [batch_size, seq_len]
                # each seq_len always different
                data_pad = pad_sequence(
                    [torch.from_numpy(np.array(x)).long() for x in batch],
                    True, self.pad_index)
                data_pad = data_pad.t().contiguous()
                # xs_pad, ys_pad: [max_seq_len, batch_size]
                # max_seq_len is the maximum length in current batch
                xs_pad = data_pad[:-1, :]
                ys_pad = data_pad[1:, :]
                yield xs_pad, ys_pad
                batch = []

    @torch.no_grad()
    def score_sentences(self, sentences: List[str]) -> torch.tensor:
        '''
        Args:
            sentences: each element is a sentence, words seperated by whitespace
        '''
        total_loss = 0.0
        average_negative_logp = []
        for batch_input, batch_target in self.batchify_sentences(sentences):
            # batch_input: [seq_len, batch_size]
            # with contents: <bos> token_id token_id ....
            #
            # batch_target: [seq_len, batch_size]
            # with contensts: token_id token_id ... <eos>
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            batch_output = self.model(batch_input)

            prediction = batch_output.view(-1, self.ntoken)
            # target: [max_seq_len * batch_size]
            # example_1_token_1 example_2_token_1 example_3_token_1 .....
            target = batch_target.view(-1)
            loss = self.criterion(prediction, target)
            average_negative_logp.append(loss.item())
        self.reset_count_variables()
        return torch.tensor(average_negative_logp).to(self.device)

    @torch.no_grad()
    def score_word_seqs(self, word_seqs: k2.RaggedInt, doing_rescore:bool = True) -> torch.tensor:
        '''
        used when rescoring
        '''
        self.set_criterion(doing_rescore=True)
        sentences = word_seqs_to_list_str(word_seqs, self.symbol_table)
        return self.score_sentences(sentences)
