#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (author: Liyong Guo)
# Apache 2.0

import logging
import math
import torch


# references:
# https://github.com/Hiroshiba/pytorch-trainer/blob/master/pytorch_trainer/training/trainer.py
# https://github.com/espnet/espnet/blob/master/espnet/lm/pytorch_backend/lm.py
# https://github.com/Hiroshiba/pytorch-trainer/blob/master/pytorch_trainer/training/trainer.py
# https://www.jianshu.com/p/c88df856dbc8
class Trainer(object):

    def __init__(self,
                 device,
                 model=None,
                 criterion=None,
                 optimizer=None,
                 train_data_loader=None,
                 dev_data_loader=None,
                 ntokens=None,
                 batch_size=1,
                 epoch=0,
                 num_epochs=10,
                 log_interval=10,
                 writer=None):
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.ntokens = ntokens
        self.batch_size = batch_size
        self.epoch = epoch
        self.num_epochs = num_epochs
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.iterations = 0
        self.writer = writer
        self.log_interval = log_interval

    def run(self):
        for epoch in range(self.num_epochs):
            if self.train_data_loader is not None:
                self.train()

            if self.dev_data_loader is not None:
                self.eval()

            self.epoch += 1

    def train(self):
        self.model.train()
        num_total_batch = len(self.train_data_loader)
        for batch_idx, batch in enumerate(self.train_data_loader):
            batch_input, batch_target = batch
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            self.model.to(self.device)
            batch_output = self.model(batch_input)

            prediction = batch_output.view(-1, self.ntokens)
            target = torch.flatten(batch_target.transpose(0, 1))
            loss = self.criterion(prediction, target)
            self.optimizer.zero_grad()
            self.optimizer.step()

            self.writer.add_scalar('train_loss', loss, self.iterations)

            self.iterations += 1
            if batch_idx % self.log_interval == 0:
                log_str = 'TRAIN Batch {}/{} loss {:.6f} ppl {:.6f} at epoch {}'.format(
                    batch_idx, num_total_batch, loss.item(),
                    math.exp(loss.item()), self.epoch)
                logging.info(log_str)

    def eval(self):
        self.model.eval()
        total_loss = 0.0
        num_total_batch = len(self.dev_data_loader)
        for batch_idx, batch in enumerate(self.dev_data_loader):
            batch_input, batch_target = batch
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            self.model.to(self.device)
            batch_output = self.model(batch_input)

            prediction = batch_output.view(-1, self.ntokens)
            target = torch.flatten(batch_target.transpose(0, 1))
            loss = self.criterion(prediction, target)
            total_loss += loss * self.batch_size

        loss = total_loss / (num_total_batch * self.batch_size)
        ppl = math.exp(loss)
        self.writer.add_scalar('dev_ppl', ppl, self.epoch)
        log_str = 'dev loss is {:.6f} and ppl {:.6f} at epoch {}'.format(
            loss.item(), ppl, self.epoch)
        logging.info(log_str)
