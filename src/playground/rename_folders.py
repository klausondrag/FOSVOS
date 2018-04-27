from pathlib import Path

from tqdm import tqdm

path_base = Path('../results/resnet18/11/')
path_output = Path('gifs')
path_output.mkdir(parents=True, exist_ok=True)
for path_variant in tqdm(path_base.iterdir()):
    parts = path_variant.stem.split('_')
    if len(parts) == 6 and int(parts[-1]) & (int(parts[-1]) - 1) == 0:
        folder_name_new = '_'.join([parts[0], parts[1], parts[4], parts[5], parts[3], parts[2]])
        path_variant_new = path_base / folder_name_new
        if not path_variant_new.exists():
            path_variant.rename(path_variant_new)
        # path_variant.delink()
