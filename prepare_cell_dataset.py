import argparse
import csv
import io
import json
import random
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def save_sample(img, points_yx, out_root, split, stem):
    img_dir = out_root / split / 'images'
    pts_dir = out_root / split / 'points'
    img_dir.mkdir(parents=True, exist_ok=True)
    pts_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_dir / f'{stem}.png'), img)
    np.save(pts_dir / f'{stem}.npy', np.asarray(points_yx, dtype=np.float32).reshape(-1, 2))


def split_indices(n, val_ratio, seed):
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    val_n = max(1, int(round(n * val_ratio))) if n > 1 else 0
    val = set(idx[:val_n])
    return ['val' if i in val else 'train' for i in range(n)]


def centroids_from_instance_map(inst_map):
    points = []
    for inst_id in np.unique(inst_map):
        if inst_id == 0:
            continue
        ys, xs = np.where(inst_map == inst_id)
        if len(ys) == 0:
            continue
        points.append([float(ys.mean()), float(xs.mean())])
    return np.asarray(points, dtype=np.float32)


def read_image_any(path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f'Failed to read image: {path}')
    return img


def load_instance_map(path_or_file):
    arr = np.load(path_or_file, allow_pickle=True)
    if arr.shape == () and arr.dtype == object:
        obj = arr.item()
        if isinstance(obj, dict) and 'inst_map' in obj:
            return obj['inst_map']
        raise ValueError('Object label npy must contain an inst_map key')
    if arr.ndim == 3:
        return arr[..., 0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f'Unsupported label shape: {arr.shape}')


def prepare_conic(src_root, out_root, val_ratio, seed):
    src_root = Path(src_root)
    out_root = Path(out_root)
    images = np.load(src_root / 'images.npy')
    labels = np.load(src_root / 'labels.npy')
    splits = split_indices(len(images), val_ratio, seed)
    for i, (img, label, split) in enumerate(zip(images, labels, splits)):
        inst_map = label[..., 0]
        points = centroids_from_instance_map(inst_map)
        save_sample(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), points, out_root, split, f'conic_{i:05d}')
    print(f'CoNIC converted to {out_root}')


def maybe_limit(samples, max_per_split):
    return samples[:max_per_split] if max_per_split and max_per_split > 0 else samples


def prepare_conic_split(split_json, zip_root, out_root, max_per_split=0):
    split_json = Path(split_json)
    zip_root = Path(zip_root)
    out_root = Path(out_root)
    with open(split_json, encoding='utf-8') as f:
        split_data = json.load(f)

    zip_cache = {}

    def get_zip(first_dir):
        if first_dir not in zip_cache:
            zip_path = zip_root / f'{first_dir}.zip'
            if not zip_path.exists():
                raise FileNotFoundError(f'Missing zip file: {zip_path}')
            zip_cache[first_dir] = zipfile.ZipFile(zip_path)
        return zip_cache[first_dir]

    try:
        for split, samples in split_data['splits'].items():
            for sample in maybe_limit(samples, max_per_split):
                image_rel = sample['image_relpath']
                label_rel = sample['label_relpath']
                first_dir = Path(image_rel).parts[0]
                zf = get_zip(first_dir)

                img_bytes = np.frombuffer(zf.read(image_rel), dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f'Failed to decode image from {image_rel}')

                with zf.open(label_rel) as f:
                    inst_map = load_instance_map(io.BytesIO(f.read()))
                points = centroids_from_instance_map(inst_map)
                save_sample(img, points, out_root, split, sample['stem'])
    finally:
        for zf in zip_cache.values():
            zf.close()
    print(f'CoNIC split converted to {out_root}')


def points_from_monuseg_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    points = []
    for region in root.findall('.//Region'):
        coords = []
        for vertex in region.findall('.//Vertex'):
            x = float(vertex.attrib['X'])
            y = float(vertex.attrib['Y'])
            coords.append([y, x])
        if coords:
            points.append(np.asarray(coords, dtype=np.float32).mean(axis=0))
    return np.asarray(points, dtype=np.float32)


def index_monuseg_files(*roots):
    images = {}
    xmls = {}
    for root in roots:
        if not root:
            continue
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if path.suffix.lower() in IMG_EXTS:
                images[path.stem] = path
            elif path.suffix.lower() == '.xml':
                xmls[path.stem] = path
    return images, xmls


def prepare_monuseg(src_root, out_root, val_ratio, seed):
    src_root = Path(src_root)
    out_root = Path(out_root)
    image_files = sorted(p for p in src_root.rglob('*') if p.suffix.lower() in IMG_EXTS)
    pairs = []
    for img_path in image_files:
        xml_path = img_path.with_suffix('.xml')
        if not xml_path.exists():
            candidates = list(src_root.rglob(f'{img_path.stem}.xml'))
            xml_path = candidates[0] if candidates else None
        if xml_path and xml_path.exists():
            pairs.append((img_path, xml_path))

    splits = split_indices(len(pairs), val_ratio, seed)
    for i, ((img_path, xml_path), split) in enumerate(zip(pairs, splits)):
        img = cv2.imread(str(img_path))
        points = points_from_monuseg_xml(xml_path)
        save_sample(img, points, out_root, split, f'monuseg_{i:05d}_{img_path.stem}')
    print(f'MoNuSeg converted to {out_root}')


def prepare_monuseg_split(split_json, train_root, test_root, out_root, max_per_split=0):
    split_json = Path(split_json)
    out_root = Path(out_root)
    with open(split_json, encoding='utf-8') as f:
        split_data = json.load(f)

    images, xmls = index_monuseg_files(train_root, test_root)
    for split in ('train', 'val', 'test'):
        for item in maybe_limit(split_data.get(split, []), max_per_split):
            sample_id = item['id'] if isinstance(item, dict) else str(item)
            img_path = images.get(sample_id)
            xml_path = xmls.get(sample_id)
            if img_path is None or xml_path is None:
                raise FileNotFoundError(f'Missing MoNuSeg image/xml for {sample_id}')
            img = read_image_any(img_path)
            points = points_from_monuseg_xml(xml_path)
            save_sample(img, points, out_root, split, sample_id)
    print(f'MoNuSeg split converted to {out_root}')


def load_points_table(path):
    path = Path(path)
    if path.suffix.lower() == '.json':
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        points = data['points'] if isinstance(data, dict) else data
        return np.asarray(points, dtype=np.float32)
    if path.suffix.lower() == '.npy':
        return np.load(path).astype(np.float32)
    delimiter = ',' if path.suffix.lower() == '.csv' else None
    return np.loadtxt(path, delimiter=delimiter, dtype=np.float32).reshape(-1, 2)


def prepare_point_folders(src_root, out_root, val_ratio, seed):
    src_root = Path(src_root)
    out_root = Path(out_root)
    image_dir = src_root / 'images'
    point_dir = src_root / 'points'
    pairs = []
    for img_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        for suffix in ('.npy', '.json', '.csv', '.txt'):
            points_path = point_dir / f'{img_path.stem}{suffix}'
            if points_path.exists():
                pairs.append((img_path, points_path))
                break

    splits = split_indices(len(pairs), val_ratio, seed)
    for img_path, points_path, split in zip([p[0] for p in pairs], [p[1] for p in pairs], splits):
        img = cv2.imread(str(img_path))
        points = load_points_table(points_path)
        save_sample(img, points, out_root, split, img_path.stem)
    print(f'Point-folder dataset converted to {out_root}')


def load_bcdata_h5(path):
    import h5py

    with h5py.File(path, 'r') as f:
        coords_xy = np.asarray(f['coordinates'][:], dtype=np.float32).reshape(-1, 2)
    return coords_xy[:, ::-1]


def prepare_bcdata(src_root, out_root):
    src_root = Path(src_root)
    out_root = Path(out_root)
    split_map = {
        'train': 'train',
        'validation': 'val',
        'test': 'test',
    }

    for raw_split, out_split in split_map.items():
        image_dir = src_root / 'images' / raw_split
        if not image_dir.exists():
            continue

        for img_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            point_sets = []
            for label_name in ('positive', 'negative'):
                h5_path = src_root / 'annotations' / raw_split / label_name / f'{img_path.stem}.h5'
                if h5_path.exists():
                    point_sets.append(load_bcdata_h5(h5_path))
            points = np.concatenate(point_sets, axis=0) if point_sets else np.empty((0, 2), dtype=np.float32)
            img = cv2.imread(str(img_path))
            save_sample(img, points, out_root, out_split, img_path.stem)
    print(f'BCData converted to {out_root}')


def write_manifest(out_root):
    out_root = Path(out_root)
    for split in ('train', 'val', 'test'):
        rows = []
        img_dir = out_root / split / 'images'
        pts_dir = out_root / split / 'points'
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            pts_path = pts_dir / f'{img_path.stem}.npy'
            if pts_path.exists():
                rows.append({
                    'image': str(img_path.relative_to(out_root)).replace('\\', '/'),
                    'points': str(pts_path.relative_to(out_root)).replace('\\', '/'),
                })
        with open(out_root / f'{split}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'points'])
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser('Prepare pathology cell datasets for PET')
    parser.add_argument('--dataset', required=True,
                        choices=['bcdata', 'conic', 'conic_split', 'monuseg', 'monuseg_split', 'point_folders'])
    parser.add_argument('--src_root')
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--split_json')
    parser.add_argument('--zip_root')
    parser.add_argument('--train_root')
    parser.add_argument('--test_root')
    parser.add_argument('--max_per_split', default=0, type=int)
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    if args.dataset == 'bcdata':
        if not args.src_root:
            raise ValueError('--src_root is required for bcdata')
        prepare_bcdata(args.src_root, args.out_root)
    elif args.dataset == 'conic':
        if not args.src_root:
            raise ValueError('--src_root is required for conic')
        prepare_conic(args.src_root, args.out_root, args.val_ratio, args.seed)
    elif args.dataset == 'conic_split':
        if not args.split_json or not args.zip_root:
            raise ValueError('--split_json and --zip_root are required for conic_split')
        prepare_conic_split(args.split_json, args.zip_root, args.out_root, args.max_per_split)
    elif args.dataset == 'monuseg':
        if not args.src_root:
            raise ValueError('--src_root is required for monuseg')
        prepare_monuseg(args.src_root, args.out_root, args.val_ratio, args.seed)
    elif args.dataset == 'monuseg_split':
        if not args.split_json or not args.train_root:
            raise ValueError('--split_json and --train_root are required for monuseg_split')
        prepare_monuseg_split(args.split_json, args.train_root, args.test_root, args.out_root, args.max_per_split)
    else:
        if not args.src_root:
            raise ValueError('--src_root is required for point_folders')
        prepare_point_folders(args.src_root, args.out_root, args.val_ratio, args.seed)
    write_manifest(args.out_root)


if __name__ == '__main__':
    main()
