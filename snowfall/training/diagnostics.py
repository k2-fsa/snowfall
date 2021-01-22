from typing import Dict

import torch
from torch import nn


def measure_weight_norms(model: nn.Module, norm: str = 'l2') -> Dict[str, float]:
    """
    Compute the norms of the model's parameters.
    Norms where it's applicable are normalized by the number
    of weights (e.g. l1 or l2).

    :param model: a torch.nn.Module instance
    :param norm: how to compute the norm. Available values: 'l1', 'l2', 'linf'
    :return: a dict mapping from parameter's name to its norm.
    """
    with torch.no_grad():
        norms = {}
        for name, param in model.named_parameters():
            if norm == 'l1':
                val = torch.mean(torch.abs(param))
            elif norm == 'l2':
                val = torch.mean(torch.pow(param, 2))
            elif norm == 'linf':
                val = torch.max(torch.abs(param))
            else:
                raise ValueError(f"Unknown norm type: {norm}")
            norms[name] = val.item()
        return norms


def measure_semiorthogonality(model: nn.Module) -> Dict[str, float]:
    """
    Compute the semi-orthogonality objective function proposed by:

        "Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks",
        Daniel Povey, Gaofeng Cheng, Yiming Wang, Ke Li, Hainan Xu, Mahsa Yarmohamadi,
        Sanjeev Khudanpur, Interspeech 2018
    """
    with torch.no_grad():
        scores = {}
        for name, m in model.named_modules():
            if hasattr(m, 'constrain_orthonormal'):
                weight = m.state_dict()['conv.weight']
                dim = weight.shape[0]
                w = weight.reshape(dim, -1)
                P = torch.mm(w, w.t())
                scale = torch.trace(torch.mm(P, P.t()) / torch.trace(P))
                I = torch.eye(dim, dtype=P.dtype, device=P.device)
                Q = P - scale * I
                score = torch.trace(torch.mm(Q, Q.t()))
                scores[name] = score.item()
        return scores


def measure_gradient_norms(model: nn.Module, norm: str = 'l1') -> Dict[str, float]:
    """
    Compute the norms of the gradients for each of model's parameters.
    Norms where it's applicable are normalized by the number
    of weights (e.g. l1 or l2).

    :param model: a torch.nn.Module instance
    :param norm: how to compute the norm. Available values: 'l1', 'l2', 'linf'
    :return: a dict mapping from parameter's name to its gradient's norm.
    """
    norms = {}
    for name, param in model.named_parameters():
        if norm == 'l1':
            val = torch.mean(torch.abs(param.grad))
        elif norm == 'l2':
            val = torch.mean(torch.pow(param.grad, 2))
        elif norm == 'linf':
            val = torch.max(torch.abs(param.grad))
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        norms[name] = val.item()
    return norms
