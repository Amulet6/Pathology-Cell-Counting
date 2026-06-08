"""
Convert BCData HDF5 annotations to APGCC txt format.

BCData structure:
  images/{train,validation,test}/<id>.png
  annotations/{train,validation,test}/positive/<id>.h5  -> coordinates (N,2) [x,y]
  annotations/{train,validation,test}/negative/<id>.h5  -> coordinates (N,2) [x,y]

Output structure (under DATA_ROOT):
  train/<id>.png          (symlink or copy)
  test/<id>.png
  train_gt/<id>.txt       (merged positive+negative coordinates, one "x y" per line)
  test_gt/<id>.txt
  train.list              (relative_img_path  relative_gt_path)
  test.list
"""

import os
import sys
import shutil
import numpy as np
import h5py
import argparse


def convert_split(data_root, split_name, out_split, out_gt_dir, list_name):
    img_dir   = os.path.join(data_root, 'images', split_name)
    pos_dir   = os.path.join(data_root, 'annotations', split_name, 'positive')
    neg_dir   = os.path.join(data_root, 'annotations', split_name, 'negative')

    out_img_dir = os.path.join(data_root, out_split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_gt_dir, exist_ok=True)

    img_files = sorted(os.listdir(img_dir), key=lambda x: int(x.split('.')[0]))
    list_lines = []

    for fname in img_files:
        stem = fname.split('.')[0]          # e.g. "0", "100"
        src_img = os.path.join(img_dir, fname)

        # symlink image into unified split dir (avoids data duplication)
        dst_img = os.path.join(out_img_dir, fname)
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)

        # merge positive + negative coordinates
        coords = []
        for ann_dir in [pos_dir, neg_dir]:
            h5_path = os.path.join(ann_dir, f'{stem}.h5')
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as f:
                    pts = f['coordinates'][:]    # shape (N, 2), [x, y]
                    coords.append(pts)

        if coords:
            all_coords = np.concatenate(coords, axis=0)
        else:
            all_coords = np.empty((0, 2), dtype=np.int64)

        # write txt annotation
        gt_path = os.path.join(out_gt_dir, f'{stem}.txt')
        with open(gt_path, 'w') as f:
            for x, y in all_coords:
                f.write(f'{x} {y}\n')

        rel_img = os.path.join(out_split, fname)
        rel_gt  = os.path.relpath(gt_path, data_root)
        list_lines.append(f'{rel_img} {rel_gt}')

    list_path = os.path.join(data_root, list_name)
    with open(list_path, 'w') as f:
        f.write('\n'.join(list_lines) + '\n')

    print(f'[{split_name}] {len(list_lines)} samples -> {list_name}')
    return len(list_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', help='path to BCData root, e.g. /mnt/data1/llx/BCData')
    args = parser.parse_args()
    root = args.data_root

    # train split
    convert_split(root,
                  split_name='train',
                  out_split='train',
                  out_gt_dir=os.path.join(root, 'train_gt'),
                  list_name='train.list')

    # validation -> used as test (APGCC only uses train.list / test.list)
    convert_split(root,
                  split_name='validation',
                  out_split='val',
                  out_gt_dir=os.path.join(root, 'val_gt'),
                  list_name='test.list')

    print('Done. Data root:', root)


if __name__ == '__main__':
    main()
