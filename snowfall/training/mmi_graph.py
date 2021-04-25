# Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)

from typing import Iterable
from typing import List
from typing import Tuple

import k2
import torch

from .ctc_graph import build_ctc_topo
from snowfall.common import get_phone_symbols
from ..lexicon import Lexicon


def create_bigram_phone_lm(phones: List[int]) -> k2.Fsa:
    '''Create a bigram phone LM.
    The resulting FSA (P) has a start-state and a state for
    each phone 1, 2, ....; and each of the above-mentioned states
    has a transition to the state for each phone and also to the final-state.

    Caution:
      blank is not a phone.

    Args:
      A list of phone IDs.

    Returns:
      An FSA representing the bigram phone LM.
    '''
    assert 0 not in phones
    final_state = len(phones) + 1
    rules = ''
    for i in range(1, final_state):
        rules += f'0 {i} {phones[i-1]} 0.0\n'

    for i in range(1, final_state):
        for j in range(1, final_state):
            rules += f'{i} {j} {phones[j-1]} 0.0\n'
        rules += f'{i} {final_state} -1 0.0\n'
    rules += f'{final_state}'
    return k2.Fsa.from_str(rules)


class MmiTrainingGraphCompiler(object):

    def __init__(
            self,
            lexicon: Lexicon,
            device: torch.device,
            oov: str = '<UNK>'
    ):
        '''
        Args:
          L_inv:
            Its labels are words, while its aux_labels are phones.
        phones:
          The phone symbol table.
        words:
          The word symbol table.
        oov:
          Out of vocabulary word.
        '''
        self.lexicon = lexicon
        L = self.lexicon.L.to(device)

        L = k2.arc_sort(L)

        assert L.requires_grad is False

        assert oov in self.lexicon.words

        self.L = L
        self.oov_id = self.lexicon.words[oov]
        self.device = device

        phone_symbols = get_phone_symbols(self.lexicon.phones)
        phone_symbols_with_blank = [0] + phone_symbols

        ctc_topo = build_ctc_topo(phone_symbols_with_blank).to(device)
        assert ctc_topo.requires_grad is False

        self.ctc_topo_inv = k2.arc_sort(ctc_topo.invert_())

    def compile(self,
                texts: Iterable[str],
                P: k2.Fsa,
                replicate_den: bool = True) -> Tuple[k2.Fsa, k2.Fsa]:
        '''Create numerator and denominator graphs from transcripts
        and the bigram phone LM.

        Args:
          texts:
            A list of transcripts. Within a transcript, words are
            separated by spaces.
          P:
            The bigram phone LM created by :func:`create_bigram_phone_lm`.
          replicate_den:
            If True, the returned den_graph is replicated to match the number
            of FSAs in the returned num_graph; if False, the returned den_graph
            contains only a single FSA
        Returns:
          A tuple (num_graph, den_graph), where

            - `num_graph` is the numerator graph. It is an FsaVec with
              shape `(len(texts), None, None)`.

            - `den_graph` is the denominator graph. It is an FsaVec with the same
              shape of the `num_graph` if replicate_den is True; otherwise, it
              is an FsaVec containing only a single FSA.
        '''
        assert P.device == self.device

        # The basic idea is described below:
        #
        # The `num_graphs` is the result of
        #
        #       ctc_topo o P o L o linear_fsa
        #
        # where linear_fsa represents the transcript.
        #
        # There are two components in the above that are changed across calls:
        #
        #   (1) The scores of P
        #   (2) linear_fsa
        #
        # We compute HPL = compose(ctc_topo, P, L) by setting P.scores to zero
        # and save an arc_map `hpl_to_p`. In this way, we can avoid invoking
        # compose(ctc_topo, P, L) and all we need to do is to update the scores
        # of HPL with the help of `hpl_to_p`.
        #
        if not hasattr(self, 'HPL_inv_sorted'):
            # self.HPL_inv_sorted saves the pre-computed result of
            # k2.arc_sort(compose(ctc_topo, P, L).invert()).
            #
            # We assume that the structure of P is not changed across calls.
            # Only its arc scores are changed.
            with torch.no_grad():
                tmp_P = P.detach().clone()

                # We need to clear the scores of P so that self.HPL_inv_sorted
                # contains only LM scores
                tmp_P.scores.zero_()
                # Add self loops here as we will use k2.intersect_device later
                P_with_self_loops, p_self_loops_to_p = \
                        k2.add_epsilon_self_loops(tmp_P, ret_arc_map=True)

                # b_to_a_map is for the following k2.intersect_device
                b_to_a_map = torch.tensor([0], dtype=torch.int32, device=self.device)

                # k2.intersect_device accepts only FsaVecs, so we use k2.create_fsa_vec()
                # in the following. Since it converts a single Fsa to an FsaVec, the
                # arc map of k2.create_fsa_vec() below is an identity map.
                HP, _, hp_to_p_self_loops = k2.intersect_device(
                    a_fsas=k2.create_fsa_vec([self.ctc_topo_inv]),
                    b_fsas=k2.create_fsa_vec([P_with_self_loops]),
                    b_to_a_map=b_to_a_map,
                    sorted_match_a=True,
                    ret_arc_maps=True)
                # The above HP is not really an HP, we have to swap its labels and aux_labels
                #
                # The arc map of k2.Fsa.invert_() is also an identity map
                HP.invert_()

                HP_sorted, hp_sorted_to_hp = k2.arc_sort(HP, ret_arc_map=True)

                # HP converts repeated phone IDs to non-repeated phone IDs.
                #
                # HP_inv_sorted is used for the following k2.intersect_device

                # The arc map between HP and HP_inv is an identity map
                HP_inv = HP.invert()

                # Sort it as we will use sorted_match_a==True
                HP_inv_sorted, hp_inv_sorted_to_hp = k2.arc_sort(HP_inv, ret_arc_map=True)

                L_with_self_loops = k2.remove_epsilon_and_add_self_loops(self.L)
                # Now L_with_self_loops.aux_labels is of type k2.RaggedInt

                # k2.intersect_device requires an FsaVec, so ... convert it
                L_with_self_loops = k2.create_fsa_vec([L_with_self_loops])

                # TODO(fangjun): Move it to k2
                # because _k2 is supposed to be invisible to end users
                import _k2
                # No need to sort L_with_self_loops since we're using
                # _k2.intersect_device
                HPL_ragged_arc, hpl_to_hp_inv_sorted, hpl_to_l_self_loops = _k2.intersect_device(
                        a_fsas=HP_inv_sorted.arcs,
                        properties_a=HP_inv_sorted.properties,
                        b_fsas=L_with_self_loops.arcs,
                        properties_b=L_with_self_loops.properties,
                        b_to_a_map=b_to_a_map,
                        need_arc_map=True,
                        sorted_match_a=True)
                HPL = k2.Fsa(HPL_ragged_arc)


                # We need to fix the labels and aux_labels of HPL
                #
                # Remember that we inverted HP. Here we copy HP_inv_sorted.aux_labels
                # as HPL's labels
                HPL.labels = k2.index(HP_inv_sorted.aux_labels, hpl_to_hp_inv_sorted)

                # Its aux_labels is from L_with_self_loops
                HPL.aux_labels = k2.index(L_with_self_loops.aux_labels, hpl_to_l_self_loops)

                assert isinstance(HPL.aux_labels, k2.RaggedInt)

                # We care about only the attributes `labels` and `aux_labels` of HPL
                #
                # We may add other attributes here, e.g., phones, for MBR training if needed.

                # We need HPL_inv since we are going to use k2.intersect_device
                #
                # HPL.aux_labels is of type k2.RaggedInt, so the returned arc map of
                # k2.invert() is not an identity map
                HPL_inv, hpl_inv_to_hpl = k2.invert(HPL, ret_arc_map=True)

                HPL_inv_sorted, hpl_inv_sorted_to_hpl_inv = k2.arc_sort(HPL_inv, ret_arc_map=True)

                hpl_inv_sorted_to_hpl = k2.compose_arc_maps(hpl_inv_to_hpl, hpl_inv_sorted_to_hpl_inv)

                hpl_inv_sorted_to_hp_inv_sorted = k2.compose_arc_maps(hpl_to_hp_inv_sorted, hpl_inv_sorted_to_hpl)

                hpl_inv_sorted_to_hp = k2.compose_arc_maps(hp_inv_sorted_to_hp, hpl_inv_sorted_to_hp_inv_sorted)

                hpl_inv_sorted_to_p_self_loops = k2.compose_arc_maps(hp_to_p_self_loops, hpl_inv_sorted_to_hp)

                hpl_inv_sorted_to_p = k2.compose_arc_maps(p_self_loops_to_p, hpl_inv_sorted_to_p_self_loops)

                # for the numerator graph
                self.HPL_inv_sorted = HPL_inv_sorted
                self.hpl_inv_sorted_to_p = hpl_inv_sorted_to_p
                self.lm_scores = self.HPL_inv_sorted.scores.clone()

                # for the denominator graph
                hp_sorted_to_p_self_loops = k2.compose_arc_maps(hp_to_p_self_loops, hp_sorted_to_hp)
                hp_sorted_to_p = k2.compose_arc_maps(p_self_loops_to_p, hp_sorted_to_p_self_loops)

                self.HP_sorted = HP_sorted
                assert self.HP_sorted.scores.abs().sum().item() == 0
                self.hp_sorted_to_p = hp_sorted_to_p

                assert self.HPL_inv_sorted.requires_grad is False
                assert self.HP_sorted.requires_grad is False
                assert self.lm_scores.requires_grad is False

        self.HPL_inv_sorted.scores = self.lm_scores + k2.index(P.scores, self.hpl_inv_sorted_to_p)

        # When this function is executed inside `with torch.no_grad()` context,
        # The following assert will fail.
        #  assert self.HPL_inv_sorted.requires_grad == P.requires_grad

        linear_fsas = self.build_linear_fsas(texts)
        linear_fsas_with_self_loops = k2.add_epsilon_self_loops(linear_fsas)

        b_to_a_map = torch.zeros(len(texts),
                                 dtype=torch.int32,
                                 device=self.device)

        num_graphs = k2.intersect_device(self.HPL_inv_sorted,
                                         linear_fsas_with_self_loops,
                                         b_to_a_map,
                                         sorted_match_a=True)
        num_graphs = k2.invert(num_graphs)

        num_graphs = k2.arc_sort(num_graphs)

        with torch.no_grad():
            self.HP_sorted.scores = k2.index(P.scores, self.hp_sorted_to_p)

        assert self.HP_sorted.requires_grad is False

        if replicate_den:
            indexes = torch.zeros(len(texts),
                                  dtype=torch.int32,
                                  device=self.device)
            den_graphs = k2.index_fsa(self.HP_sorted, indexes)
        else:
            den_graphs = self.HP_sorted

        return num_graphs, den_graphs

    def build_linear_fsas(self, texts: List[str]) -> k2.Fsa:
        '''Convert transcript to an Fsa with the help of lexicon
        and word symbol table.

        Args:
          texts:
            Each element is a transcript containing words separated by spaces.
            For instance, one of the elements may be 'HELLO SNOWFALL', which
            contains two words.

        Returns:
          Return an FSA (FsaVec) corresponding to the transcript. Its `labels` are
          word IDs.
        '''
        word_ids_list = []
        for text in texts:
            word_ids = []
            for word in text.split(' '):
                if word in self.lexicon.words:
                    word_ids.append(self.lexicon.words[word])
                else:
                    word_ids.append(self.oov_id)
            word_ids_list.append(word_ids)

        fsa = k2.linear_fsa(word_ids_list, self.device)
        return fsa
