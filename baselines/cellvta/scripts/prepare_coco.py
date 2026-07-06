#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
from PIL import Image


def symlink_or_replace(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src, dst.parent)
    os.symlink(rel, dst)


def clip(v, lo, hi):
    return max(lo, min(v, hi))


def square_ann(x, y, radius, w, h):
    x1 = clip(int(round(x - radius)), 0, w - 1)
    y1 = clip(int(round(y - radius)), 0, h - 1)
    x2 = clip(int(round(x + radius)), 0, w - 1)
    y2 = clip(int(round(y + radius)), 0, h - 1)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    seg = [[x1, y1, x1 + bw, y1, x1 + bw, y1 + bh, x1, y1 + bh]]
    return [x1, y1, bw, bh], seg


def coco_base():
    return {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'nucleus', 'supercategory': 'cell'}],
    }


def add_image(coco, image_id, file_name, width, height):
    coco['images'].append({
        'id': image_id,
        'file_name': file_name,
        'width': width,
        'height': height,
    })


def add_ann(coco, ann_id, image_id, bbox, seg, area):
    coco['annotations'].append({
        'id': ann_id,
        'image_id': image_id,
        'category_id': 1,
        'bbox': [float(x) for x in bbox],
        'segmentation': seg,
        'area': float(area),
        'iscrowd': 0,
    })


def load_h5_points(path: Path):
    with h5py.File(path, 'r') as f:
        pts = f['coordinates'][:]
    return pts.tolist()


def convert_bcdata(src_root: Path, out_root: Path, radius: int, seed: int):
    img_root = src_root / 'BCData' / 'images'
    ann_root = src_root / 'BCData' / 'annotations'
    split_map = {
        'train': 'train',
        'validation': 'val',
        'test': 'test',
    }
    for src_split, out_split in split_map.items():
        images = sorted((img_root / src_split).glob('*.png'))
        coco = coco_base()
        img_out_dir = out_root / 'images' / out_split
        img_out_dir.mkdir(parents=True, exist_ok=True)
        ann_id = 1
        for image_id, img_path in enumerate(images, start=1):
            with Image.open(img_path) as im:
                w, h = im.size
            symlink_or_replace(img_path, img_out_dir / img_path.name)
            add_image(coco, image_id, f'{out_split}/{img_path.name}', w, h)
            stem = img_path.stem
            p_pos = ann_root / src_split / 'positive' / f'{stem}.h5'
            p_neg = ann_root / src_split / 'negative' / f'{stem}.h5'
            ann_path = p_pos if p_pos.exists() else p_neg
            if ann_path.exists():
                pts = load_h5_points(ann_path)
                for x, y in pts:
                    bbox, seg = square_ann(x, y, radius, w, h)
                    add_ann(coco, ann_id, image_id, bbox, seg, bbox[2] * bbox[3])
                    ann_id += 1
        (out_root / 'annotations').mkdir(parents=True, exist_ok=True)
        with open(out_root / 'annotations' / f'{out_split}.json', 'w', encoding='utf-8') as f:
            json.dump(coco, f)

    train = json.loads((out_root / 'annotations' / 'train.json').read_text(encoding='utf-8'))
    val = json.loads((out_root / 'annotations' / 'val.json').read_text(encoding='utf-8'))
    test = json.loads((out_root / 'annotations' / 'test.json').read_text(encoding='utf-8'))
    trainval = coco_base()
    trainval['images'] = train['images'] + val['images']
    trainval['annotations'] = train['annotations'] + val['annotations']
    with open(out_root / 'annotations' / 'trainval.json', 'w', encoding='utf-8') as f:
        json.dump(trainval, f)

    # convenience aliases for code that expects train/val/test folders
    for split in ('train', 'val', 'test'):
        (out_root / 'images' / split).mkdir(parents=True, exist_ok=True)


def parse_monu_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    regions = root.find('.//Regions')
    anns = []
    if regions is None:
        return anns
    for reg in list(regions):
        verts = reg.find('Vertices')
        if verts is None:
            continue
        pts = [(float(v.attrib['X']), float(v.attrib['Y'])) for v in list(verts)]
        if len(pts) < 3:
            continue
        anns.append(pts)
    return anns


def polygon_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    return [x1, y1, bw, bh], [[float(v) for p in poly for v in p]]


def convert_monuseg(src_root: Path, out_root: Path, seed: int, test_ratio: float):
    img_root = src_root / 'MoNuSeg 2018 Training Data' / 'Tissue Images'
    ann_root = src_root / 'MoNuSeg 2018 Training Data' / 'Annotations'
    images = sorted(img_root.glob('*.tif'))
    rng = random.Random(seed)
    rng.shuffle(images)
    n_test = max(1, int(round(len(images) * test_ratio)))
    test_imgs = images[:n_test]
    trainval_imgs = images[n_test:]
    n_val = max(1, int(round(len(trainval_imgs) * 0.2))) if len(trainval_imgs) > 1 else 0
    val_imgs = trainval_imgs[:n_val]
    train_imgs = trainval_imgs[n_val:]
    splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs, 'trainval': trainval_imgs}

    ann_dir = out_root / 'annotations'
    img_dir = out_root / 'images'
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    for split, split_imgs in splits.items():
        coco = coco_base()
        split_dir = img_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        ann_id = 1
        for image_id, img_path in enumerate(split_imgs, start=1):
            with Image.open(img_path) as im:
                w, h = im.size
            symlink_or_replace(img_path, split_dir / img_path.name)
            add_image(coco, image_id, f'{split}/{img_path.name}', w, h)
            xml_path = ann_root / f'{img_path.stem}.xml'
            for poly in parse_monu_xml(xml_path):
                bbox, seg = polygon_bbox(poly)
                area = bbox[2] * bbox[3]
                add_ann(coco, ann_id, image_id, bbox, seg, area)
                ann_id += 1
        with open(ann_dir / f'{split}.json', 'w', encoding='utf-8') as f:
            json.dump(coco, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['bcdata', 'monuseg', 'all'], default='all')
    p.add_argument('--src-root', type=Path, default=Path('/data/zju-151/liyixin/project/pathology-cell-counting/datasets'))
    p.add_argument('--out-root', type=Path, default=Path('/data/zju-151/liyixin/project/pathology-cell-counting/datasets/processed'))
    p.add_argument('--radius', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-ratio', type=float, default=0.2)
    args = p.parse_args()

    if args.dataset in ('bcdata', 'all'):
        convert_bcdata(args.src_root / 'BCData', args.out_root / 'bcdata', args.radius, args.seed)
    if args.dataset in ('monuseg', 'all'):
        convert_monuseg(args.src_root / 'MoNuSeg', args.out_root / 'monuseg', args.seed, args.test_ratio)

if __name__ == '__main__':
    main()
