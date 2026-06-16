from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=Path, required=True)
    parser.add_argument('--output_root', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    images = np.load(args.input_root / 'images.npy', mmap_mode='r')
    labels = np.load(args.input_root / 'labels.npy', mmap_mode='r')
    patch_info = pd.read_csv(args.input_root / 'patch_info.csv')
    counts = pd.read_csv(args.input_root / 'counts.csv')

    out_img = args.output_root / 'images'
    out_lab = args.output_root / 'labels'
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    total = images.shape[0]
    if args.limit > 0:
        total = min(total, args.limit)

    manifest = []
    for i in range(total):
        name = str(patch_info.iloc[i, 0])
        image = images[i]
        inst_map = labels[i, :, :, 0].astype(np.int64)
        type_map = labels[i, :, :, 1].astype(np.int64)

        Image.fromarray(image.astype(np.uint8)).save(out_img / f'{name}.png')
        np.save(out_lab / f'{name}.npy', inst_map)

        item = {
            'name': name,
            'instances': int(inst_map.max()),
            'types_present': [int(x) for x in np.unique(type_map).tolist() if int(x) != 0],
            'counts': {k: int(v) for k, v in counts.iloc[i].to_dict().items()},
        }
        manifest.append(item)
        if i < 5 or (i + 1) % 500 == 0:
            print(i + 1, name, 'instances=', item['instances'])

    with open(args.output_root / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    counts.to_csv(args.output_root / 'counts.csv', index=False)
    patch_info.to_csv(args.output_root / 'patch_info.csv', index=False)

    print('done', total)


if __name__ == '__main__':
    main()
