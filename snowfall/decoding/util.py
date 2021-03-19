# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

import k2
import torch


def get_log_probs(phone_fsas: k2.Fsa, nnet_output: torch.Tensor,
                  len_per_path: torch.Tensor):
    '''
    Args:
      phone_fsas:
        An FsaVec with three axes [path][state][arc]
      nnet_output:
        A 3-D torch.Tensor (num_path, seq_len, num_phones+1)
      len_per_path:
        A 1-D torch.Tensor of shape (num_path)
    '''
    assert len(phone_fsas.shape) == 3
    offset = phone_fsas.arcs.row_splits(2)[phone_fsas.arcs.row_splits(
        1).long()]
    expected_len_per_path = offset[1:] - offset[:-1]
    assert torch.all(torch.eq(len_per_path, expected_len_per_path.cpu()))

    phone_fsas = phone_fsas.to('cpu')
    nnet_output = nnet_output.to('cpu')
    assert phone_fsas.arcs.dim0() == nnet_output.shape[0]

    probs = []
    num_paths = phone_fsas.arcs.dim0()
    for i in range(num_paths):
        this_fsa = k2.index(phone_fsas, torch.tensor([i], dtype=torch.int32))
        len_this_fsa = len_per_path[i]
        this_fsa_nnet_output = nnet_output[i]
        this_prob = []
        labels = this_fsa.labels
        for idx, row in enumerate(this_fsa_nnet_output):
            if idx >= len_this_fsa:
                break
            this_prob.append(row[labels[idx]].item())
        probs.append(this_prob)

    assert len(probs) == num_paths
    ans = k2.ragged.create_ragged2(probs)

    return ans
