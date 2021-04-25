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
        num_graphs, den_graphs = self.graph_compiler.compile(
            texts, self.P, replicate_den=False)

        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)

        device = num_graphs.device

        num_fsas = num_graphs.shape[0]
        assert dense_fsa_vec.dim0() == num_fsas

        assert den_graphs.shape[0] == 1

        # the aux_labels of num_graphs is k2.RaggedInt
        # but it is torch.Tensor for den_graphs.
        #
        # The following converts den_graphs.aux_labels
        # from torch.Tensor to k2.RaggedInt so that
        # we can use k2.append() later
        if not isinstance(den_graphs.aux_labels, k2.RaggedInt):
            # This only needs to be done once if `den_graphs`
            # is shared across calls.
            den_graphs.convert_attr_to_ragged_(name='aux_labels')

        num_den_graphs = k2.cat([num_graphs, den_graphs])

        # NOTE: The a_to_b_map in k2.intersect_dense must be sorted
        # so the following reorders num_den_graphs.

        # [0, 1, 2, ... ]
        num_graphs_indexes = torch.arange(num_fsas, dtype=torch.int32)

        # [num_fsas, num_fsas, num_fsas, ... ]
        den_graphs_indexes = torch.tensor([num_fsas] * num_fsas,
                                          dtype=torch.int32)

        # [0, num_fsas, 1, num_fsas, 2, num_fsas, ... ]
        num_den_graphs_indexes = torch.stack(
            [num_graphs_indexes, den_graphs_indexes]).t().reshape(-1).to(device)

        num_den_reordered_graphs = k2.index(num_den_graphs, num_den_graphs_indexes)

        # [[0, 1, 2, ...]]
        a_to_b_map = torch.arange(num_fsas, dtype=torch.int32).reshape(1, -1)

        # [[0, 1, 2, ...]] -> [0, 0, 1, 1, 2, 2, ... ]
        a_to_b_map = a_to_b_map.repeat(2, 1).t().reshape(-1).to(device)

        num_den_lats = k2.intersect_dense(num_den_reordered_graphs,
                                          dense_fsa_vec,
                                          output_beam=10.0,
                                          a_to_b_map=a_to_b_map)

        num_den_tot_scores = num_den_lats.get_tot_scores(
            log_semiring=True, use_double_scores=True)

        num_tot_scores = num_den_tot_scores[::2]
        den_tot_scores = num_den_tot_scores[1::2]

        tot_scores = num_tot_scores - self.den_scale * den_tot_scores
        tot_score, tot_frames, all_frames = get_tot_objf_and_num_frames(
            tot_scores,
            supervision_segments[:, 2]
        )
        return tot_score, tot_frames, all_frames
