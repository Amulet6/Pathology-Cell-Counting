"""
Convert MoNuSeg dataset to APGCC txt format.

MoNuSeg structure:
  Training:
    "MoNuSeg 2018 Training Data/Tissue Images/<id>.tif"
    "MoNuSeg 2018 Training Data/Annotations/<id>.xml"
  Test:
    "MoNuSegTestData/<id>.tif"
    "MoNuSegTestData/<id>.xml"

Each XML contains <Region> elements whose <Vertices> define a nucleus polygon.
We compute the centroid of each polygon as the annotation point.

Output (under DATA_ROOT):
  train/<id>.png
  test/<id>.png
  train_gt/<id>.txt     "x y" per nucleus
  test_gt/<id>.txt
  train.list
  test.list
"""

import os
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import cv2


def parse_xml(xml_path):
    """Return list of (cx, cy) centroids from a MoNuSeg XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    centroids = []
    for region in root.iter('Region'):
        xs, ys = [], []
        for vertex in region.iter('Vertex'):
            xs.append(float(vertex.attrib['X']))
            ys.append(float(vertex.attrib['Y']))
        if xs:
            centroids.append((np.mean(xs), np.mean(ys)))
    return centroids


def process_split(img_dir, ann_dir, out_img_dir, out_gt_dir, data_root, list_name):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_gt_dir, exist_ok=True)

    tif_files = sorted(f for f in os.listdir(img_dir) if f.endswith('.tif'))
    list_lines = []

    for fname in tif_files:
        stem = os.path.splitext(fname)[0]
        img_src = os.path.join(img_dir, fname)
        xml_path = os.path.join(ann_dir, stem + '.xml')

        # save image as PNG
        img = cv2.imread(img_src)
        out_img_path = os.path.join(out_img_dir, stem + '.png')
        cv2.imwrite(out_img_path, img)

        # parse annotations
        centroids = parse_xml(xml_path) if os.path.exists(xml_path) else []
        gt_path = os.path.join(out_gt_dir, stem + '.txt')
        with open(gt_path, 'w') as f:
            for cx, cy in centroids:
                f.write(f'{cx:.2f} {cy:.2f}\n')

        split = os.path.basename(out_img_dir)
        rel_img = os.path.join(split, stem + '.png')
        rel_gt  = os.path.relpath(gt_path, data_root)
        list_lines.append(f'{rel_img} {rel_gt}')

    list_path = os.path.join(data_root, list_name)
    with open(list_path, 'w') as f:
        f.write('\n'.join(list_lines) + '\n')

    print(f'[{list_name}] {len(list_lines)} images saved.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', help='output root, e.g. /mnt/data1/llx/MoNuSegdata')
    parser.add_argument('--train-img', required=True, help='path to "MoNuSeg 2018 Training Data/Tissue Images"')
    parser.add_argument('--train-ann', required=True, help='path to "MoNuSeg 2018 Training Data/Annotations"')
    parser.add_argument('--test-dir',  required=True, help='path to MoNuSegTestData (contains .tif and .xml)')
    args = parser.parse_args()

    root = args.data_root
    os.makedirs(root, exist_ok=True)

    process_split(
        img_dir=args.train_img,
        ann_dir=args.train_ann,
        out_img_dir=os.path.join(root, 'train'),
        out_gt_dir=os.path.join(root, 'train_gt'),
        data_root=root,
        list_name='train.list',
    )

    process_split(
        img_dir=args.test_dir,
        ann_dir=args.test_dir,
        out_img_dir=os.path.join(root, 'test'),
        out_gt_dir=os.path.join(root, 'test_gt'),
        data_root=root,
        list_name='test.list',
    )

    print('Done. Data root:', root)


if __name__ == '__main__':
    main()
