# Copyright (c)  2020-2021  Xiaomi Corporation (authors: Daniel Povey
#                                                        Haowen Qiu
#                                                        Fangjun Kuang)
# Apache 2.0

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from snowfall.models import AcousticModel
from snowfall.training.diagnostics import measure_weight_norms


class Tdnn1a(AcousticModel):
    """
    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
    """

    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 subsampling_factor: int = 3) -> None:
        super(Tdnn1a, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        self.tdnn = nn.Sequential(
            nn.Conv1d(in_channels=num_features,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(in_channels=500,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(in_channels=500,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(in_channels=500,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(in_channels=500,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(in_channels=500,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(
                in_channels=500,
                out_channels=500,
                kernel_size=3,
                stride=self.
                subsampling_factor,  # <---- stride=3: subsampling_factor!
                padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(in_channels=500,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(in_channels=500,
                      out_channels=2000,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=2000, affine=False),
            nn.Conv1d(in_channels=2000,
                      out_channels=2000,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=2000, affine=False),
            nn.Conv1d(in_channels=2000,
                      out_channels=num_classes,
                      kernel_size=1,
                      stride=1,
                      padding=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
          x:
            Tensor of dimension (batch_size, num_features, input_length).

        Returns:
          A torch.Tensor of dimension (batch_size, number_of_classes, input_length).
        '''

        x = self.tdnn(x)
        x = F.log_softmax(x, dim=1)
        return x

    def write_tensorboard_diagnostics(self,
                                      tb_writer: SummaryWriter,
                                      global_step: Optional[int] = None):
        tb_writer.add_scalars('train/weight_l2_norms',
                              measure_weight_norms(self, norm='l2'),
                              global_step=global_step)
        tb_writer.add_scalars('train/weight_max_norms',
                              measure_weight_norms(self, norm='linf'),
                              global_step=global_step)


class Tdnn2aEmbedding(nn.Module):

    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.num_features = num_classes
        self.num_classes = num_classes
        self.subsampling_factor = 1
        # left context: 1 + 1 + 3 + 3 = 8
        # right context: 1 + 1 + 3 + 3 = 8
        self.tdnn = nn.Sequential(
            nn.Conv1d(in_channels=num_features,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            #
            nn.Conv1d(in_channels=500,
                      out_channels=500,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=500, affine=False),
            #
            nn.Conv1d(in_channels=500,
                      out_channels=2000,
                      kernel_size=3,
                      stride=1,
                      dilation=3,
                      padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=2000, affine=False),
            #
            nn.Conv1d(in_channels=2000,
                      out_channels=2000,
                      kernel_size=3,
                      stride=1,
                      dilation=3,
                      padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=2000, affine=False),
            #
            nn.Conv1d(in_channels=2000,
                      out_channels=num_classes,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
          x:
            A tensor of shape (N, num_features, T)
        Returns:
          Return a tensor of shape (N, num_classes, T)
        '''
        x = self.tdnn(x)
        x = F.log_softmax(x, dim=1)
        return x

    def write_tensorboard_diagnostics(self,
                                      tb_writer: SummaryWriter,
                                      global_step: Optional[int] = None):
        tb_writer.add_scalars('train/tdnn2a_weight_l2_norms',
                              measure_weight_norms(self, norm='l2'),
                              global_step=global_step)
        tb_writer.add_scalars('train/tdnn2a_weight_max_norms',
                              measure_weight_norms(self, norm='linf'),
                              global_step=global_step)
