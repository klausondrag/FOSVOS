from pathlib import Path
import itertools

import click
from tqdm import tqdm
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
        for file in tqdm(list(p.iterdir())):
            image = cv2.imread(str(file))

            assert len(image.shape) == 3
            channel_dimension = np.where(np.asarray(image.shape) == 3)[0][0]
            channels = list(range(3))
            del channels[channel_dimension]

            image_mean = image.mean(axis=channels[1]).mean(axis=channels[0])
            mean += image_mean
            n_images += 1

    mean /= n_images
    log.info('Calculated mean: {}'.format(str(mean)))


@cli.command()
@click.pass_context
def filter(ctx: click.core.Context) -> None:
    dataset_dir = ctx.obj['dataset_dir']
    dataset_dir = Path(dataset_dir)

    source_path = dataset_dir / 'source'
    annotations_path = dataset_dir / 'foreground_annotations'
    foreground_path = dataset_dir / 'foreground'
    foreground_path.mkdir(exist_ok=True)
    for annotation_file in tqdm(list(annotations_path.iterdir())):
        annotation_image = cv2.imread(str(annotation_file))
        color_file_name = annotation_file.stem + '.jpg'
        source_file = source_path / color_file_name
        source_image = cv2.imread(str(source_file))

        foreground_image = np.where((annotation_image >= 1), source_image, annotation_image)
        foreground_file = foreground_path / color_file_name
        cv2.imwrite(str(foreground_file), foreground_image)
        # show_image(filtered_image)


@cli.command()
@click.pass_context
def overlay(ctx: click.core.Context) -> None:
    dataset_dir = ctx.obj['dataset_dir']
    dataset_dir = Path(dataset_dir)

    background_path = dataset_dir / 'background'
    foreground_path = dataset_dir / 'foreground'
    foreground_annotations_path = dataset_dir / 'foreground_annotations'
    output_path = dataset_dir / 'images'
    output_path.mkdir(exist_ok=True)
    output_annotations_path = dataset_dir / 'annotations'
    output_annotations_path.mkdir(exist_ok=True)
    pairs = list(itertools.product(background_path.iterdir(), foreground_path.iterdir()))
    for index, (background_file, foreground_file) in enumerate(tqdm(pairs)):
        background_image = cv2.imread(str(background_file))
        foreground_image = cv2.imread(str(foreground_file))
        annotation_file = foreground_annotations_path / '{}.png'.format(foreground_file.stem)
        annotation_image = cv2.imread(str(annotation_file))

        scale_factor = 1 - np.random.ranf() / 2

        output_annotation_image = cv2.resize(annotation_image, dsize=(0, 0), fx=scale_factor, fy=scale_factor,
                                             interpolation=cv2.INTER_AREA)
        output_annotation_file = output_annotations_path / '{}.png'.format(index)
        cv2.imwrite(str(output_annotation_file), output_annotation_image)
        # show_image(output_annotation_image)

        foreground_image = cv2.resize(foreground_image, dsize=(0, 0), fx=scale_factor, fy=scale_factor,
                                      interpolation=cv2.INTER_AREA)
        x_offset = y_offset = 0
        y1, y2 = y_offset, y_offset + foreground_image.shape[0]
        x1, x2 = x_offset, x_offset + foreground_image.shape[1]
        alpha_s = (output_annotation_image.astype(float) / 255).mean(axis=2)
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            background_image[y1:y2, x1:x2, c] = (alpha_s * foreground_image[:, :, c] +
                                                 alpha_l * background_image[y1:y2, x1:x2, c])
        output_image = background_image

        output_file = output_path / '{}.jpg'.format(index)
        cv2.imwrite(str(output_file), output_image)
        # show_image(output_image)


def show_image(image: np.ndarray) -> None:
    from matplotlib import pyplot as plt
    plt.figure()
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img)
    plt.show(block=True)


if __name__ == '__main__':
    cli(obj={})
