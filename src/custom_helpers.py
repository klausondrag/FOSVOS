from pathlib import Path

import click
import cv2

from util.logger import get_logger

log = get_logger(__file__)


@click.group()
@click.option('--dataset-dir', '-d', type=click.Path(file_okay=False), default='/home/klaus/dev/datasets/Me')
@click.pass_context
def cli(ctx: click.core.Context, dataset_dir: str) -> None:
    ctx.obj['dataset_dir'] = dataset_dir


@cli.command()
@click.pass_context
def mean(ctx: click.core.Context) -> None:
    dataset_dir = ctx.obj['dataset_dir']
    dataset_dir = Path(dataset_dir)

    for directory in ['Images']:
        p = dataset_dir / directory
        for file in p.iterdir():
            image = cv2.imread(str(file))
            log.info(image.mean())


if __name__ == '__main__':
    cli(obj={})
