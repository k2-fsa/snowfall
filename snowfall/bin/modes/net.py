# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

from pathlib import Path

import click
import k2
import lhotse
import torch

from .cli_base import cli


@cli.group()
def net():
    '''
    Neural network tools in snowfall.
    '''
    pass


@net.command()
@click.option('-m',
              '--model',
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help='Path to Torch Scripted module')
@click.option('-f',
              '--feats',
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help='Path to Featureset manifest')
@click.option('-l',
              '--max-duration',
              default=200,
              type=int,
              required=True,
              help='max duration in seconds in a batch')
@click.option('-i',
              '--device-id',
              default=0,
              type=int,
              required=True,
              help='-1 to use CPU. Otherwise, it is the GPU device ID')
@click.option('-o',
              '--output-dir',
              type=click.Path(dir_okay=True),
              required=True,
              help='Output directory')
def compute_post(model: str, feats: str, max_duration: int, device_id: int,
                 output_dir: str):
    '''Compute posteriors given a model and a FeatureSet.
    '''
    scripted_model = torch.jit.load(model)
    scripted_model.eval()

    if device_id < 0:
        print('Use CPU')
        device = torch.device('cpu')
    else:
        print(f'Use GPU {device_id}')
        device = torch.device('cuda', device_id)

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
            print(f'Processing batch {i}')

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
    print('Done!')
    print(f"Files are saved to the directory: '{output_dir}'")
