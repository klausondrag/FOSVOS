from pathlib import Path

import numpy as np
from tqdm import tqdm
import imageio
from moviepy.editor import ImageSequenceClip
import click


# necessary because at the time of writing moviepy expects to be 3 dimensional
def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        width, height = image.shape
        rgb = np.empty((width, height, 3), dtype=np.uint8)
        rgb[:, :, :] = image[:, :, None]
        return rgb
    else:
        return image


def dir_to_images(path: Path):
    files = path.iterdir()
    files = map(str, files)
    files = sorted(files)
    files = map(imageio.imread, files)
    files = map(convert_to_rgb, files)
    files = list(files)
    return files


def generate_gif(path_input: Path, path_output_file: Path, output_format: str) -> None:
    if not path_output_file.exists():
        try:
            files = dir_to_images(path_input)
            clip = ImageSequenceClip(files, fps=16)

            if output_format == 'gif':
                clip.write_gif(str(path_output_file), fps=16)
            elif output_format == 'mp4':
                clip.write_videofile(str(path_output_file), fps=16)
            else:
                raise Exception('Unknown format: ', output_format)
        except Exception as e:
            print('Skipped ', str(path_output_file), 'because', str(e))


@click.command()
@click.option('--path-base', type=str, default='../results/resnet18/11/')
@click.option('--path-output', type=str, default='../results/gifs')
@click.option('--sequence-name', type=str, default='blackswan')
@click.option('--output-format', type=click.Choice(['gif', 'mp4']), default='gif')
def convert_folder(path_base, path_output, sequence_name, output_format):
    path_base = Path(path_base)
    path_output = Path(path_output) / sequence_name
    path_output.mkdir(parents=True, exist_ok=True)
    for path_variant in sorted(path_base.iterdir()):
        generate_gif(path_variant / sequence_name,
                     path_output / (path_variant.stem + '.' + output_format),
                     output_format)


if __name__ == '__main__':
    convert_folder()
