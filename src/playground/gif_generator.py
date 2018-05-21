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


sequences_val = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows',
                 'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'horsejump-high', 'kite-surf',
                 'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']

sequences_train = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'dance-jump',
                   'dog-agility', 'drift-turn', 'elephant', 'flamingo', 'hike', 'hockey', 'horsejump-low',
                   'kite-walk', 'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike',
                   'paragliding', 'rhino', 'rollerblade', 'scooter-gray', 'soccerball', 'stroller', 'surf',
                   'swing', 'tennis', 'train']

sequences_all = list(set(sequences_train + sequences_val))


@click.command()
@click.option('--path-base-input', type=str, default='../results/resnet18/11')
@click.option('--path-base-output', type=str, default='../results/gifs')
@click.option('--output-format', type=click.Choice(['gif', 'mp4']), default='gif')
@click.option('--mode', type=click.Choice(['prune', 'mimic']), default='prune')
def convert_folder(path_base_input, path_base_output, output_format, mode):
    path_base_input = Path(path_base_input) / mode
    path_base_output = Path(path_base_output) / mode

    for sequence_name in sequences_all:
        path_output = path_base_output / sequence_name
        path_output.mkdir(parents=True, exist_ok=True)
        for path_variant in sorted(path_base_input.iterdir()):
            if mode == 'mimic':
                path_input = path_variant / '300' / sequence_name
            elif mode == 'prune':
                path_input = path_variant / sequence_name
            else:
                raise Exception('Unknown mode')

            if path_input.exists():
                generate_gif(path_input, path_output / (path_variant.name + '.' + output_format), output_format)


if __name__ == '__main__':
    convert_folder()
