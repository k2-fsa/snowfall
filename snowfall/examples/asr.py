import logging
import re
from typing import List, Optional, Sequence, Tuple, Union

import k2
import torch

from lhotse import CutSet, Fbank, FbankConfig
from lhotse.cut import AnyCut, Cut, MixedCut
from lhotse.dataset import K2SpeechRecognitionDataset, OnTheFlyFeatures
from lhotse.supervision import AlignmentItem
from lhotse.utils import fastcopy
from snowfall.common import Pathlike, average_checkpoint, get_texts
from snowfall.lexicon import Lexicon
from snowfall.models.conformer import Conformer
from snowfall.objectives import encode_supervisions
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler, create_bigram_phone_lm


class ASR:
    """
    This class is a high-level wrapper for K2 acoustic models that simplifies inference:
    reading models, computing posteriors, decoding, alignments, etc.

    Currently it will only work with the Conformer model with a very specific HMM topology.
    It could be the basis for a more generic entry point to Snow(Ice?)fall.
    """

    def __init__(
            self,
            lang_dir: Pathlike,
            scripted_model_path: Optional[Pathlike] = None,
            model_dir: Optional[Pathlike] = None,
            average_epochs: Sequence[int] = (7, 8, 9),
            device: torch.device = 'cpu',
            sampling_rate: int = 16000,
    ):
        if isinstance(device, str):
            self.device = torch.device(device)

        self.sampling_rate = sampling_rate
        self.extractor = Fbank(FbankConfig(num_mel_bins=80))
        self.lexicon = Lexicon(lang_dir)
        phone_ids = self.lexicon.phone_symbols()
        self.P = create_bigram_phone_lm(phone_ids)

        if model_dir is not None:
            # Read model from regular checkpoints, assume it's a Conformer
            self.model = Conformer(
                num_features=80,
                num_classes=len(phone_ids) + 1,
                num_decoder_layers=0
            )
            self.P.scores = torch.zeros_like(self.P.scores)
            self.model.P_scores = torch.nn.Parameter(self.P.scores.clone(), requires_grad=False)
            average_checkpoint(
                filenames=[model_dir / f'epoch-{n}.pt' for n in average_epochs],
                model=self.model
            )
        elif scripted_model_path is not None:
            # Read model from a serialized TorchScript module, no assumptions needed
            self.model = torch.jit.load(scripted_model_path)
        else:
            raise ValueError("One of scripted_model_path or model_dir needs to be provided.")

        # Freeze the params by default.
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.compiler = MmiTrainingGraphCompiler(lexicon=self.lexicon, device=self.device)
        self.HLG = k2.Fsa.from_dict(torch.load(lang_dir / 'HLG.pt')).to(self.device)

    def compute_features(self, cuts: Union[AnyCut, CutSet]) -> torch.Tensor:
        if isinstance(cuts, (Cut, MixedCut)):
            cuts = CutSet.from_cuts([cuts])
        assert cuts[0].sampling_rate == self.sampling_rate, f'{cuts[0].sampling_rate} != {self.sampling_rate}'
        otf = OnTheFlyFeatures(self.extractor)
        # feats: (batch, seq_len, n_feats)
        feats, _ = otf(cuts)
        return feats

    def compute_posteriors(self, cuts: Union[AnyCut, CutSet]) -> torch.Tensor:
        """
        Run the forward pass of the acoustic model and return a tensor representing a batch of phone posteriorgrams.
        """
        # Extract feats
        # (batch, seq_len, num_feats)
        if isinstance(cuts, (Cut, MixedCut)):
            cuts = CutSet.from_cuts([cuts])
        assert cuts[0].sampling_rate == self.sampling_rate, f'{cuts[0].sampling_rate} != {self.sampling_rate}'
        otf = OnTheFlyFeatures(self.extractor)
        # feats: (batch, seq_len, n_feats)
        feats, _ = otf(cuts)
        # feats: (batch, n_feats, seq_len)
        feats = feats.permute(0, 2, 1)

        # Compute AM posteriors
        # posteriors: (batch, n_phones, ~seq_len / 4)
        posteriors, _, _ = self.model(feats)
        # returns: (batch, ~seq_len / 4, n_phones)
        return posteriors.permute(0, 2, 1)

    def decode(self, cuts: Union[AnyCut, CutSet]) -> List[Tuple[List[str], List[str]]]:
        """
        Perform decoding with an n-gram language model (HLG graph).
        Doesn't support rescoring at this time.
        """
        word_results = []
        # Hacky way to get batch quickly... we may need to improve on this.
        batch = K2SpeechRecognitionDataset(
            cuts,
            input_strategy=OnTheFlyFeatures(self.extractor),
            check_inputs=False
        )[list(cuts.ids)]
        features = batch['inputs'].permute(0, 2, 1).to(self.device)  # (B, T, F) -> (B, F, T)
        supervision_segments, texts = encode_supervisions(batch['supervisions'])

        # Forward pass through the acoustic model
        posteriors, _, _ = self.model(features)
        posteriors = posteriors.permute(0, 2, 1)  # (B, F, T) -> (B, T, F)

        # Wrapping into k2 "dense FSA" (representing PPG as a dense graph)
        dense_fsa_vec = k2.DenseFsaVec(posteriors, supervision_segments)

        # The actual decoding starts here:
        # First, we intersect the HLG and the PPG
        # with default pruning/beam search params from snowfall
        # The result is a batch of graphs (lattices)
        lattices = k2.intersect_dense_pruned(self.HLG, dense_fsa_vec, 20.0, 8, 30, 10000)
        # ... then we find the shortest paths in the lattices ...
        best_paths = k2.shortest_path(lattices, use_double_scores=True)
        # ... and convert them to words with a convenience wrapper from snowfall
        hyps = get_texts(best_paths, torch.arange(len(texts)))

        # Here we read out the words from the best path graphs
        for i in range(len(texts)):
            hyp_words = [self.lexicon.words.get(x) for x in hyps[i]]
            ref_words = texts[i].split(' ')
            word_results.append((ref_words, hyp_words))
        return word_results

    def align(self, cuts: Union[AnyCut, CutSet]) -> torch.Tensor:
        """
        Perform forced alignment and return a tensor that represents a batch of frame-level alignments:
        >>> alignments = torch.tensor([
        ...     [0, 0, 0, 1, 57, 57, 35, 35, 35, ...],
        ...     [...],
        ...     ...
        ... ])

        :return: an int32 tensor with shape ``(batch_size, num_frames)``.
        """
        # Extract feats
        # (batch, seq_len, num_feats)
        if isinstance(cuts, (Cut, MixedCut)):
            cuts = CutSet.from_cuts([cuts])
        assert cuts[0].sampling_rate == self.sampling_rate, f'{cuts[0].sampling_rate} != {self.sampling_rate}'

        cuts = cuts.map_supervisions(self.normalize_text)

        otf = OnTheFlyFeatures(self.extractor)
        feats, _ = otf(cuts)
        feats = feats.permute(0, 2, 1)
        texts = [' '.join(s.text for s in cut.supervisions) for cut in cuts]

        # Compute AM posteriors
        # (batch, seq_len ~/ 4, num_phones)
        posteriors, _, _ = self.model(feats)
        # Note: we are using "dummy" supervisions so that the aligner also considers
        # the padding area. We can adjust that behaviour if needed by passing actual
        # supervision segments, but then we will have a ragged tensor (will need to
        # pad the alignments themselves).
        sups = self.dummy_supervisions(feats)
        posteriors_fsa = k2.DenseFsaVec(posteriors.permute(0, 2, 1), sups)

        # Intersection with ground truth transcript graphs
        num, den = self.compiler.compile(texts, self.P)
        alignment = k2.intersect_dense(num, posteriors_fsa, output_beam=10.0)
        best_path = k2.shortest_path(alignment, use_double_scores=True)

        # Retrieve sequences of phone IDs per frame
        # (batch, seq_len ~/ 4) -- dtype int32 (num phone labels)
        frame_labels = torch.stack([best_path[i].labels[:-1] for i in range(best_path.shape[0])])
        return frame_labels

    def align_ctm(self, cuts: Union[CutSet, AnyCut]) -> List[List[AlignmentItem]]:
        """
        Perform forced alignment and parse the phones into a CTM-like format:
            >>> [[0.0, 0.12, 'SIL'], [0.12, 0.2, 'AH0'], ...]
        """
        # TODO: I am not sure that this method is extracting the alignment 100% correctly:
        #       need to revise...
        # TODO: when K2/Snowfall has a standard way of indicating what is silence,
        #       or we update the model, update the constants below.
        EPS = 0
        SIL = 1
        non_speech = {EPS, SIL}

        def to_s(n: int) -> float:
            FRAME_SHIFT = 0.04  # 0.01 * 4 subsampling
            return round(n * FRAME_SHIFT, ndigits=3)

        if isinstance(cuts, (Cut, MixedCut)):
            cuts = CutSet.from_cuts([cuts])

        # Uppercase and remove punctuation
        cuts = cuts.map_supervisions(self.normalize_text)
        alignments = self.align(cuts).tolist()

        ctm_alis = []
        for cut, alignment in zip(cuts, alignments):
            # First we determine the silence regions at the beginning and the end:
            # we assume that every SIL and <eps> before the first phone, and after the last phone,
            # are representing silence.
            first_speech_idx = [idx for idx, s in enumerate(alignment) if s not in non_speech][0]
            last_speech_idx = [idx for idx, s in reversed(list(enumerate(alignment))) if s not in non_speech][0]
            speech_ali = alignment[first_speech_idx: last_speech_idx]
            ctm_ali = [
                AlignmentItem(start=0.0, duration=to_s(first_speech_idx), symbol=self.lexicon.phones[SIL])
            ]

            # Then, we iterate over the speech region: since the K2 model uses 2-state HMM
            # topology that allows blank (<eps>) to follow a phone symbol, we treat <eps>
            # as continuation of the "previous" phone.
            # TODO: I think this implementation is wrong in that it merges repeating phones...
            #       Will fix.
            # TODO: I think it could be simplified by using some smart semi-ring and FSA operations...
            start = first_speech_idx
            prev_s = speech_ali[0]
            curr_s = speech_ali[0]
            cntr = 1
            for s in speech_ali[1:]:
                curr_s = s if s != EPS else curr_s
                if curr_s != prev_s:
                    ctm_ali.append(
                        AlignmentItem(start=to_s(start), duration=to_s(cntr), symbol=self.lexicon.phones[prev_s])
                    )
                    start = start + cntr
                    prev_s = curr_s
                    cntr = 1
                else:
                    cntr += 1
            if cntr:
                ctm_ali.append(
                    AlignmentItem(start=to_s(start), duration=to_s(cntr), symbol=self.lexicon.phones[prev_s])
                )

            speech_end_timestamp = to_s(last_speech_idx)
            if speech_end_timestamp > cut.duration:
                logging.warning(f"speech_end_timestamp <= cut.duration. Skipping cut {cut.id}")
                ctm_alis.append(None)
                continue

            ctm_ali.append(
                AlignmentItem(
                    start=speech_end_timestamp,
                    duration=round(cut.duration - speech_end_timestamp, ndigits=8),
                    symbol=self.lexicon.phones[SIL])
            )
            ctm_alis.append(ctm_ali)

        return ctm_alis

    def plot_alignments(self, cut: AnyCut):
        import matplotlib.pyplot as plt
        feats = self.compute_features(cut)
        phone_ids = self.align(cut)
        fig, axes = plt.subplots(2, squeeze=True, sharex=True, sharey=True, figsize=(10, 14))
        axes[0].imshow(feats[0])
        axes[1].imshow(torch.nn.functional.one_hot(phone_ids.to(torch.int64)).T)
        return fig, axes

    def plot_posteriors(self, cut: AnyCut):
        import matplotlib.pyplot as plt
        feats = self.compute_features(cut)
        posteriors = self.compute_posteriors(cut)
        fig, axes = plt.subplots(2, squeeze=True, sharex=True, sharey=True, figsize=(10, 14))
        axes[0].imshow(feats[0])
        axes[1].imshow(posteriors[0].exp())
        return fig, axes

    @staticmethod
    def dummy_supervisions(feats):
        def size_after_conv(size, num_layers=2):
            for i in range(num_layers):
                size = (size - 1) // 2
            return size

        return torch.tensor(
            [[
                i,
                size_after_conv(2, num_layers=2),
                size_after_conv(feats.shape[2] - 2, num_layers=2)
            ] for i in range(feats.size(0))],
            dtype=torch.int32
        ).clamp(min=0)

    @staticmethod
    def normalize_text(supervision):
        text = re.sub(r'[^\w\s]', '', supervision.text.upper())
        return fastcopy(supervision, text=text)
