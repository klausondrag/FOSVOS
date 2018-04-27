from pathlib import Path

import numpy as np
from tqdm import tqdm
import imageio
from moviepy.editor import ImageSequenceClip


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


def generate_gif(path_input: Path, path_output_file: Path) -> None:
    if not path_output_file.exists():
        try:
            files = dir_to_images(path_input)
            clip = ImageSequenceClip(files, fps=16)
            # clip.write_videofile(str(path_output_file), fps=16)
            clip.write_gif(str(path_output_file), fps=16)
        except Exception as e:
            print('Skipped ', str(path_output_file), 'because', str(e))


path_base = Path('../results/resnet18/11/')
path_output = Path('gifs')
path_output.mkdir(parents=True, exist_ok=True)
for path_variant in path_base.iterdir():
    generate_gif(path_variant / 'blackswan', path_output / (path_variant.stem + '.gif'))
