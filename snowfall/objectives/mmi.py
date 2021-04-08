from typing import List, Tuple

import torch
from torch import nn

import k2

from snowfall.objectives.common import get_tot_objf_and_num_frames
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler


class LFMMILoss(nn.Module):
    """
    Computes Lattice-Free Maximum Mutual Information (LFMMI) loss.

    TODO: more detailed description
    """
    def __init__(
            self,
            graph_compiler: MmiTrainingGraphCompiler,
            P: k2.Fsa,
            den_scale: float = 1.0,
    ):
        super().__init__()
        self.graph_compiler = graph_compiler
        self.P = P
        self.den_scale = den_scale

    def forward(
            self,
            nnet_output: torch.Tensor,
            texts: List,
            supervision_segments: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int]:
        num, den = self.graph_compiler.compile(texts, self.P)
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)

        num = k2.intersect_dense(num, dense_fsa_vec, 10.0)
        den = k2.intersect_dense(den, dense_fsa_vec, 10.0)

        num_tot_scores = num.get_tot_scores(
            log_semiring=True,
            use_double_scores=True
        )
        den_tot_scores = den.get_tot_scores(
            log_semiring=True,
            use_double_scores=True
        )
        tot_scores = num_tot_scores - self.den_scale * den_tot_scores
        tot_score, tot_frames, all_frames = get_tot_objf_and_num_frames(
            tot_scores,
            supervision_segments[:, 2]
        )
        return tot_score, tot_frames, all_frames
