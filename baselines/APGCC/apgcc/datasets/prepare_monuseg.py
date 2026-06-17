"""
Convert MoNuSeg 2018 to APGCC txt format with a reproducible 30/7/14 split.

Split is defined by ``monuseg_split.json`` (next to this script): the exact TCGA
ids for train(30) / val(7) / test(14). See that file for the rationale (held-out
rare-organ validation; organ per image derived from the TCGA TSS code). train+val
ids live in "MoNuSeg 2018 Training Data" (37 images); test ids live in
"MoNuSegTestData" (14 images).

MoNuSeg structure:
  Training pool (train + val):
    "MoNuSeg 2018 Training Data/Tissue Images/<id>.tif"
    "MoNuSeg 2018 Training Data/Annotations/<id>.xml"
  Test:
    "MoNuSegTestData/<id>.tif"
    "MoNuSegTestData/<id>.xml"

Each XML contains <Region> elements whose <Vertices> define a nucleus polygon.
We compute the centroid (mean of polygon vertices) as the annotation point.

Output (under DATA_ROOT):
  train/<id>.png   val/<id>.png   test/<id>.png
  train_gt/<id>.txt  val_gt/<id>.txt  test_gt/<id>.txt    "x y" per nucleus
  train.list  val.list  test.list                          "<split>/<id>.png <split>_gt/<id>.txt"
"""

import os
import json
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


def process_split(ids, img_dir, ann_dir, out_img_dir, out_gt_dir, data_root, list_name):
    """Convert the given ids (found under img_dir/ann_dir) into <split>/ + <split>_gt/ + list."""
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_gt_dir, exist_ok=True)

    split = os.path.basename(out_img_dir)
    list_lines = []

    for stem in ids:
        img_src = os.path.join(img_dir, stem + '.tif')
        xml_path = os.path.join(ann_dir, stem + '.xml')
        if not os.path.exists(img_src):
            raise FileNotFoundError('image not found: %s' % img_src)

        # save image as PNG
        img = cv2.imread(img_src)
        out_img_path = os.path.join(out_img_dir, stem + '.png')
        cv2.imwrite(out_img_path, img)

        # parse annotations -> centroids
        centroids = parse_xml(xml_path) if os.path.exists(xml_path) else []
        gt_path = os.path.join(out_gt_dir, stem + '.txt')
        with open(gt_path, 'w') as f:
            for cx, cy in centroids:
                f.write('%.2f %.2f\n' % (cx, cy))

        rel_img = os.path.join(split, stem + '.png')
        rel_gt = os.path.relpath(gt_path, data_root)
        list_lines.append('%s %s' % (rel_img, rel_gt))

    list_path = os.path.join(data_root, list_name)
    with open(list_path, 'w') as f:
        f.write('\n'.join(list_lines) + '\n')

    print('[%s] %d images saved.' % (list_name, len(list_lines)))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', help='output root, e.g. /data1/llx/MoNuSegdata')
    parser.add_argument('--train-img', required=True, help='path to "MoNuSeg 2018 Training Data/Tissue Images"')
    parser.add_argument('--train-ann', required=True, help='path to "MoNuSeg 2018 Training Data/Annotations"')
    parser.add_argument('--test-dir',  required=True, help='path to MoNuSegTestData (contains .tif and .xml)')
    parser.add_argument('--split-json', default=os.path.join(here, 'monuseg_split.json'),
                        help='split definition (default: monuseg_split.json next to this script)')
    args = parser.parse_args()

    with open(args.split_json) as f:
        split = json.load(f)
    train_ids = [x['id'] for x in split['train']]
    val_ids   = [x['id'] for x in split['val']]
    test_ids  = [x['id'] for x in split['test']]
    print('Split: %d train / %d val / %d test (from %s)' %
          (len(train_ids), len(val_ids), len(test_ids), os.path.basename(args.split_json)))

    root = args.data_root
    os.makedirs(root, exist_ok=True)

    # train and val both come from the 37-image training pool
    process_split(train_ids, args.train_img, args.train_ann,
                  os.path.join(root, 'train'), os.path.join(root, 'train_gt'), root, 'train.list')
    process_split(val_ids, args.train_img, args.train_ann,
                  os.path.join(root, 'val'), os.path.join(root, 'val_gt'), root, 'val.list')
    # test comes from the official MoNuSegTestData (images and xml in the same dir)
    process_split(test_ids, args.test_dir, args.test_dir,
                  os.path.join(root, 'test'), os.path.join(root, 'test_gt'), root, 'test.list')

    print('Done. Data root:', root)


if __name__ == '__main__':
    main()
