from pathlib import Path

import click
import numpy as np
import cv2

from util.logger import get_logger

log = get_logger(__file__)


@click.group()
@click.option('--dataset-dir', '-d', type=click.Path(file_okay=False), default='/home/klaus/dev/datasets/Me1080')
@click.pass_context
def cli(ctx: click.core.Context, dataset_dir: str) -> None:
    ctx.obj['dataset_dir'] = dataset_dir


@cli.command()
@click.pass_context
def mean(ctx: click.core.Context) -> None:
    dataset_dir = ctx.obj['dataset_dir']
    dataset_dir = Path(dataset_dir)

    mean = np.zeros(3)
    n_images = 0
    for directory in ['background', 'source']:
        p = dataset_dir / directory
        for file in p.iterdir():
            image = cv2.imread(str(file))

            assert len(image.shape) == 3
            channel_dimension = np.where(np.asarray(image.shape) == 3)[0][0]
            channels = list(range(3))
            del channels[channel_dimension]

            image_mean = image.mean(axis=channels[1]).mean(axis=channels[0])
            mean += image_mean
            n_images += 1

    mean /= n_images
    log.info('Found n images: {}'.format(n_images))
    log.info('Calculated mean: {}'.format(str(mean)))


@cli.command()
@click.pass_context
def filter(ctx: click.core.Context) -> None:
    dataset_dir = ctx.obj['dataset_dir']
    dataset_dir = Path(dataset_dir)

    n_images = 0
    source_path = dataset_dir / 'source'
    annotations = dataset_dir / 'annotations'
    for annotation_file in annotations.iterdir():
        image = cv2.imread(str(annotation_file))

        n_images += 1

    log.info('Found n images: {}'.format(n_images))


if __name__ == '__main__':
    cli(obj={})
