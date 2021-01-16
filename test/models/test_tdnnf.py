import math

import pytest
import torch

from snowfall.models.tdnnf import FactorizedTDNN, Tdnnf1a, _constrain_orthonormal_internal

torch.manual_seed(20200130)


def test_constrain_orthonormal():
    def compute_loss(M):
        P = torch.mm(M, M.t())
        P_PT = torch.mm(P, P.t())

        trace_P = torch.trace(P)
        trace_P_P = torch.trace(P_PT)

        scale = torch.sqrt(trace_P_P / trace_P)

        identity = torch.eye(P.size(0), dtype=P.dtype, device=P.device)
        Q = P / (scale * scale) - identity
        loss = torch.norm(Q, p='fro')  # Frobenius norm

        return loss

    w = torch.randn(6, 8) * 10

    loss = []
    loss.append(compute_loss(w))

    for i in range(15):
        w = _constrain_orthonormal_internal(w)
        loss.append(compute_loss(w))

    for i in range(1, len(loss)):
        assert loss[i - 1] > loss[i]

    # TODO(fangjun): draw the loss using matplotlib
    #  print(loss)

    model = FactorizedTDNN(dim=1024,
                           bottleneck_dim=128,
                           kernel_size=3,
                           subsampling_factor=1)
    loss = []

    for m in model.modules():
        if hasattr(m, 'constrain_orthonormal'):
            m.constrain_orthonormal()

    loss.append(
        compute_loss(model.linear.conv.state_dict()['weight'].reshape(128, -1)))
    for i in range(5):
        for m in model.modules():
            if hasattr(m, 'constrain_orthonormal'):
                m.constrain_orthonormal()
        loss.append(
            compute_loss(model.linear.conv.state_dict()['weight'].reshape(
                128, -1)))

    for i in range(1, len(loss)):
        assert loss[i - 1] > loss[i]


def test_factorized_tdnn():
    N = 1
    T = 10
    C = 4

    # case 0: kernel_size == 1, subsampling_factor == 1
    model = FactorizedTDNN(dim=C,
                           bottleneck_dim=2,
                           kernel_size=1,
                           subsampling_factor=1)
    x = torch.arange(N * T * C).reshape(N, C, T).float()
    y = model(x)
    assert y.size(2) == T

    # case 1: kernel_size == 3, subsampling_factor == 1
    model = FactorizedTDNN(dim=C,
                           bottleneck_dim=2,
                           kernel_size=3,
                           subsampling_factor=1)
    y = model(x)
    assert y.size(2) == T - 2

    # case 2: kernel_size == 1, subsampling_factor == 3
    model = FactorizedTDNN(dim=C,
                           bottleneck_dim=2,
                           kernel_size=1,
                           subsampling_factor=3)
    y = model(x)
    assert y.size(2) == math.ceil(math.ceil((T - 3)) - 3)


@pytest.mark.parametrize('seq_len', [126, 127, 128, 129, 130, 131])
def test_subsampling_matched_lengths(seq_len):
    num_features = 4
    num_classes = 7
    subsampled_seq_len = math.ceil(seq_len / 3)
    model = Tdnnf1a(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=16,
        bottleneck_dim=2,
        prefinal_bottleneck_dim=4,
    )
    inputs = torch.randn(1, num_features, seq_len)
    outputs = model(inputs)
    assert len(outputs.shape) == 3
    assert outputs.shape[0] == 1
    assert outputs.shape[1] == subsampled_seq_len
    assert outputs.shape[2] == num_classes
