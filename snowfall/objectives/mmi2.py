from typing import List, Optional, Tuple

import torch
from torch import nn

import k2

from snowfall.objectives.common2 import get_tot_objf_and_num_frames
from snowfall.training.mmi_graph2 import MmiTrainingGraphCompiler


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

    def forward(self,
                nnet_output: torch.Tensor,
                texts: List[str],
                supervision_segments: torch.Tensor,
                ret_den_lats: bool = False
               ) -> Tuple[torch.Tensor, int, int, Optional[k2.Fsa]]:
        '''
        Args:
          nnet_output:
            A 3-D tensor of shape (N, T, F). It is passed to
            :func:`k2.DenseFsaVec`, so it represents log_probs,
            from `log_softmax()`.
          texts:
            A list of str. Each list item contains a transcript.
            A transcript consists of space(s) separated words.
            An example transcript looks like 'hello snowfall'.
            An example texts is given below:

                ['hello k2', 'hello snowfall']

          supervision_segments:
            A 2-D tensor that will be passed to :func:`k2.DenseFsaVec`.
            See :func:`k2.DenseFsaVec` for its format.
          ret_den_lats:
            True to also return the resulting denominator lattice.
        Returns:
          Return a tuple containing 6 entries:

            - A tensor with only one element containing the loss

            - Number of frames that contributes to the returned loss.
              Note that frames of sequences that result in an infinity
              loss are not counted.

            - Number of frames used in the computation.

            - The denominator lattice if ret_den_lats is True.
              Otherwise, it is None.
            -
        '''
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
        # we can use k2.cat() later
        den_graphs.convert_attr_to_ragged_(name='aux_labels')

        num_den_graphs = k2.cat([num_graphs, den_graphs])

        # NOTE: The a_to_b_map in k2.intersect_dense must be sorted
        # so the following reorders num_den_graphs. It also replicates
        # den_graphs.

        # [0, 1, 2, ... ]
        num_graphs_indexes = torch.arange(num_fsas, dtype=torch.int32)

        # [num_fsas, num_fsas, num_fsas, ... ]
        den_graphs_indexes = torch.tensor([num_fsas] * num_fsas,
                                          dtype=torch.int32)

        # [0, num_fsas, 1, num_fsas, 2, num_fsas, ... ]
        num_den_graphs_indexes = torch.stack(
            [num_graphs_indexes,
             den_graphs_indexes]).t().reshape(-1).to(device)

        num_den_reordered_graphs = k2.index(num_den_graphs,
                                            num_den_graphs_indexes)
        # Now num_den_reordered_graphs contains
        # [num_graph0, den_graph0, num_graph1, den_graph1, ... ]

        # [[0, 1, 2, ...]]
        a_to_b_map = torch.arange(num_fsas, dtype=torch.int32).reshape(1, -1)

        # [[0, 1, 2, ...]] -> [0, 0, 1, 1, 2, 2, ... ]
        a_to_b_map = a_to_b_map.repeat(2, 1).t().reshape(-1).to(device)

        num_den_lats = k2.intersect_dense(num_den_reordered_graphs,
                                          dense_fsa_vec,
                                          output_beam=10.0,
                                          a_to_b_map=a_to_b_map)
        # num_den_lats contains
        # [num_lats0, den_lats0, num_lats1, den_lats1, ... ]

        num_den_tot_scores = num_den_lats.get_tot_scores(
            log_semiring=True, use_double_scores=True)

        num_tot_scores = num_den_tot_scores[::2]
        den_tot_scores = num_den_tot_scores[1::2]

        tot_scores = num_tot_scores - self.den_scale * den_tot_scores
        tot_score, tot_frames, all_frames = get_tot_objf_and_num_frames(
            tot_scores, supervision_segments[:, 2])
        if ret_den_lats:
            # [1, 3, 5, ... ]
            den_lats_indexes = torch.arange(start=1,
                                            end=(2 * num_fsas),
                                            step=2,
                                            dtype=torch.int32,
                                            device=device)
            with torch.no_grad():
                den_lats = k2.index(num_den_lats, den_lats_indexes)

            assert den_lats.requires_grad is False
        else:
            den_lats = None
            num_den_reordered_graphs = None
            a_to_b_map = None

        # TODO(fangjun): return a dict
        return tot_score, tot_frames, all_frames, den_lats, num_den_reordered_graphs, a_to_b_map
