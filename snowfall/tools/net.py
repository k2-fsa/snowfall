# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

from pathlib import Path
from typing import Any
from typing import Dict

import sys

import k2
import lhotse
import torch

from snowfall.common import invert_permutation
from snowfall.decoding.graph import load_or_compile_HLG

from .ali import get_phone_alignment


def compute_post(model: str,
                 feats: str,
                 output_dir: str,
                 device: torch.device,
                 max_duration: int = 200) -> None:
    '''Compute posteriors given a model and a FeatureSet.

    Caution:
      It supports only the transformer/conformer model
      in snowfall due to its implicit assumption of
      the signature of the `forward()` function of the model.

    Args:
      model:
        Filename to the torch scripted model.
      feats:
        Filename to the FeatureSet manifest.
      output_dir:
        Output directory.
      device:
        It indicates which device to use for running the network.
      max_duration:
        Maximum duration in seconds in a batch.
    Returns:
      Return None.
    '''
    scripted_model = torch.jit.load(model)
    scripted_model.eval()

    scripted_model.to(device)

    cuts = lhotse.load_manifest(feats)

    dataset = lhotse.dataset.K2SpeechRecognitionDataset(cuts, return_cuts=True)

    sampler = lhotse.dataset.SingleCutSampler(cuts, max_duration=max_duration)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=None,
                                             sampler=sampler,
                                             num_workers=1)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    storage_path = output_dir / 'posts'

    posts_writer = lhotse.NumpyFilesWriter(storage_path=storage_path)
    ans_cuts = []

    for i, batch in enumerate(dataloader):
        if i % 10 == 0:
            print(f'Processing batch {i}', file=sys.stderr)

        feature = batch['inputs'].to(device)
        feature_lens = batch['inputs_lens']

        supervisions = batch['supervisions']

        cut = supervisions['cut']

        # Delete 'text' and 'cut' since they are not tensors
        # and torch script does not support them.
        del supervisions['text']
        del supervisions['cut']

        # Caution: The following code is specific to the current
        # transformer/conformer model in snowfall
        #
        # There are two inputs:
        #  (1) features, with shape [N, C, T]
        #  (2) supervisions
        #
        # Its output is of shape [N, C, T]
        #
        # The model uses a subsampling factor of 4

        # at entry, feature is [N, T, C]
        feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]

        with torch.no_grad():
            # nnet_output is [N, C, T]
            nnet_output, _, _ = scripted_model(feature, supervisions)

        nnet_output = nnet_output.permute(0, 2, 1)
        # now nnet_output is [N, T, C]

        nnet_output_lens = ((feature_lens - 1) // 2 - 1) // 2
        nnet_output_lens = nnet_output_lens.tolist()

        for posts, c, num_frames in zip(nnet_output, cut, nnet_output_lens):
            posts = posts[:num_frames].cpu().numpy()
            posteriors = lhotse.save_posteriors(posts,
                                                subsampling_factor=4,
                                                storage=posts_writer)

            ans_cuts.append(c.attach_posteriors(posteriors))

    ans_cutset = lhotse.CutSet.from_cuts(ans_cuts)
    ans_cutset.to_json(output_dir / f'cuts_post.json.gz')
    print('Done!', file=stderr)
    print(f"Files are saved to the directory: '{output_dir}'", file=stderr)


def _decode_one_batch(batch: Dict[str, Any], HLG: k2.Fsa,
                      output_beam_size: float) -> k2.Fsa:
    '''Decode one batch to get 1-best path.

    Args:
      batch:
        It is the return value from K2SpeechRecognitionDataset().
        See the doc of that class for the content in the batch.
      HLG:
        The decoding graph.
      output_beam_size:
        Size of the beam for pruning during decoding.
    Returns:
      Return a 1-best path for each utterance in the batch.
    '''
    device = HLG.device

    nnet_output = batch['inputs'].to(device)
    supervisions = batch['supervisions']

    supervision_segments = torch.stack([
        supervisions['sequence_idx'],
        supervisions['start_frame'],
        supervisions['num_frames'],
    ], 1).to(torch.int32)

    # Sort the supervision_segments by num_frames.
    # The sort is required by the subsequent
    # operation: k2.intersect_dense_pruned
    indices = torch.argsort(supervision_segments[:, 2], descending=True)
    supervision_segments = supervision_segments[indices]

    # We assume that all utterances within the batch
    # have the same subsampling factor.
    sf = supervisions['cut'][0].posts.subsampling_factor
    dense_fsa_vec = k2.DenseFsaVec(nnet_output,
                                   supervision_segments,
                                   allow_truncate=sf - 1)
    lattices = k2.intersect_dense_pruned(HLG, dense_fsa_vec, 20.0,
                                         output_beam_size, 30, 10000)

    # best_paths is an FsaVec. Its shape is (batch_size, None, None)
    best_paths = k2.shortest_path(lattices, use_double_scores=True)

    # NOTE: We sorted the supervision_segments before, so we need
    # to undo that operation.
    inverted_indices = invert_permutation(indices).to(device=device,
                                                      dtype=torch.int32)
    best_paths = k2.index_fsa(best_paths, inverted_indices)
    return best_paths


def decode(lang_dir: str,
           posts: str,
           output_dir: str,
           device: torch.device,
           max_duration: int = 200,
           output_beam_size: float = 8.0):
    HLG = load_or_compile_HLG(lang_dir)

    HLG = HLG.to(device)
    HLG.aux_labels = k2.ragged.remove_values_eq(HLG.aux_labels, 0)
    HLG.requires_grad_(False)

    cuts = lhotse.load_manifest(posts)
    dataset = lhotse.dataset.K2SpeechRecognitionDataset(
        cuts,
        input_strategy=lhotse.dataset.PrecomputedPosteriors(),
        return_cuts=True)
    sampler = lhotse.dataset.SingleCutSampler(cuts, max_duration=max_duration)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=None,
                                             sampler=sampler,
                                             num_workers=1)

    ans = {}
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 10 == 0:
            print(f'Processing batch {batch_idx}')

        best_paths = _decode_one_batch(batch=batch,
                                       HLG=HLG,
                                       output_beam_size=output_beam_size)

        phone_alignment = get_phone_alignment(best_paths)
        # TODO(fangjun): implement `get_word_alignment()`.

        cut = batch['supervisions']['cut']
        for ali, c in zip(phone_alignment, cut):
            # We use the cut id to identify the utterance
            ans[c.id] = ali

    output_dir = Path(output_dir)
    out_file = output_dir / 'ali.pt'
    torch.save(ans, out_file)

    print(f'Saved to {out_file}', file=sys.stderr)
