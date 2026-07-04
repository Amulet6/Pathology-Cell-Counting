from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_regions(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    regions = root.find('.//Regions')
    polys = []
    if regions is None:
        return polys
    for region in list(regions):
        if region.tag != 'Region':
            continue
        vertices = region.find('Vertices')
        if vertices is None:
            continue
        pts = []
        for v in list(vertices):
            try:
                x = float(v.attrib['X'])
                y = float(v.attrib['Y'])
            except Exception:
                continue
            pts.append((x, y))
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def rasterize_instance_mask(size, polygons):
    mask = np.zeros((size[1], size[0]), dtype=np.int32)
    for idx, poly in enumerate(polygons, start=1):
        img = Image.new('L', size, 0)
        draw = ImageDraw.Draw(img)
        draw.polygon(poly, outline=idx, fill=idx)
        poly_mask = np.array(img, dtype=np.int32)
        mask = np.where(poly_mask > 0, idx, mask)
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=Path, required=True)
    parser.add_argument('--output_root', type=Path, required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    image_dir = args.input_root / 'MoNuSeg 2018 Training Data' / 'Tissue Images'
    anno_dir = args.input_root / 'MoNuSeg 2018 Training Data' / 'Annotations'
    out_img = args.output_root / 'images'
    out_lab = args.output_root / 'labels'
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    items = sorted(image_dir.glob('*.tif'))
    if args.limit > 0:
        items = items[:args.limit]

    manifest = []
    for idx, img_path in enumerate(items):
        xml_path = anno_dir / f'{img_path.stem}.xml'
        if not xml_path.exists():
            continue
        img = Image.open(img_path).convert('RGB')
        polygons = parse_regions(xml_path)
        if not polygons:
            continue
        mask = rasterize_instance_mask(img.size, polygons)
        out_name = img_path.stem
        img.save(out_img / f'{out_name}.png')
        np.save(out_lab / f'{out_name}.npy', mask)
        manifest.append({'image': f'{out_name}.png', 'label': f'{out_name}.npy', 'instances': int(mask.max())})
        print(idx, out_name, 'instances=', int(mask.max()))

    with open(args.output_root / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
