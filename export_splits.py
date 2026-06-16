import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np


def sha1_text(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def read_pet_manifest(pet_root, split):
    manifest = pet_root / f'{split}.csv'
    if not manifest.exists():
        img_dir = pet_root / split / 'images'
        pts_dir = pet_root / split / 'points'
        if not img_dir.exists() or not pts_dir.exists():
            return []
        rows = []
        for img_path in sorted(img_dir.iterdir()):
            pts_path = pts_dir / f'{img_path.stem}.npy'
            if pts_path.exists():
                rows.append({
                    'image': str(img_path.relative_to(pet_root)).replace('\\', '/'),
                    'points': str(pts_path.relative_to(pet_root)).replace('\\', '/'),
                })
        return rows
    with open(manifest, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def sample_id(dataset, image_path):
    stem = Path(image_path).stem
    if dataset.lower() == 'conic':
        match = re.search(r'conic_(\d+)$', stem)
        return str(int(match.group(1))) if match else stem
    if dataset.lower() == 'monuseg':
        match = re.search(r'monuseg_\d+_(.+)$', stem)
        return match.group(1) if match else stem
    return stem


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['id', 'image', 'points']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_pet_splits(args):
    pet_root = Path(args.pet_root)
    out_dir = Path(args.out_dir) / args.dataset
    summary = {
        'dataset': args.dataset,
        'source': str(pet_root),
        'mode': 'export_pet',
        'splits': {},
    }

    for split in args.splits:
        rows_raw = read_pet_manifest(pet_root, split)
        rows = []
        ids = []
        for row in rows_raw:
            sid = sample_id(args.dataset, row['image'])
            ids.append(sid)
            rows.append({'id': sid, 'image': row['image'], 'points': row['points']})
        write_lines(out_dir / f'{split}_ids.txt', ids)
        write_csv(out_dir / f'{split}.csv', rows)
        summary['splits'][split] = {
            'count': len(ids),
            'sha1': sha1_text('\n'.join(ids)),
        }

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Exported PET splits to {out_dir}')
    print(json.dumps(summary['splits'], indent=2))


def split_indices(n, ratios, seed):
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError('Ratios must sum to 1.0')
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    train_end = int(round(n * ratios[0]))
    val_end = train_end + int(round(n * ratios[1]))
    return {
        'train': sorted(idx[:train_end]),
        'val': sorted(idx[train_end:val_end]),
        'test': sorted(idx[val_end:]),
    }


def make_conic_index_splits(args):
    src_root = Path(args.src_root)
    images = np.load(src_root / 'images.npy', mmap_mode='r')
    ratios = [float(x) for x in args.ratios.split(',')]
    splits = split_indices(len(images), ratios, args.seed)
    out_dir = Path(args.out_dir) / args.dataset
    summary = {
        'dataset': args.dataset,
        'source': str(src_root),
        'mode': 'make_conic_indices',
        'seed': args.seed,
        'ratios': ratios,
        'splits': {},
    }

    for split, ids_int in splits.items():
        ids = [str(i) for i in ids_int]
        write_lines(out_dir / f'{split}_ids.txt', ids)
        rows = [{'id': sid, 'image': f'images.npy[{sid}]', 'points': f'labels.npy[{sid}]'} for sid in ids]
        write_csv(out_dir / f'{split}.csv', rows)
        summary['splits'][split] = {
            'count': len(ids),
            'sha1': sha1_text('\n'.join(ids)),
        }

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Generated CoNIC index splits to {out_dir}')
    print(json.dumps(summary['splits'], indent=2))


def main():
    parser = argparse.ArgumentParser('Export fixed train/val/test splits for shared experiments')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_export = sub.add_parser('export-pet', help='Export splits from an existing PET-format dataset')
    p_export.add_argument('--dataset', required=True)
    p_export.add_argument('--pet_root', required=True)
    p_export.add_argument('--out_dir', default='splits')
    p_export.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    p_export.set_defaults(func=export_pet_splits)

    p_conic = sub.add_parser('make-conic', help='Generate fixed index splits for raw CoNIC images.npy/labels.npy')
    p_conic.add_argument('--dataset', default='CoNIC')
    p_conic.add_argument('--src_root', required=True)
    p_conic.add_argument('--out_dir', default='splits')
    p_conic.add_argument('--ratios', default='0.7,0.15,0.15', help='train,val,test ratios')
    p_conic.add_argument('--seed', default=42, type=int)
    p_conic.set_defaults(func=make_conic_index_splits)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
