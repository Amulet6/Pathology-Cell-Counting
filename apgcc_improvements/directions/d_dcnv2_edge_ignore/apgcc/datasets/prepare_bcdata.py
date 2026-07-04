"""
Convert BCData HDF5 annotations to APGCC txt format.

BCData structure:
  images/{train,validation,test}/<id>.png
  annotations/{train,validation,test}/positive/<id>.h5  -> coordinates (N,2) [x,y]
  annotations/{train,validation,test}/negative/<id>.h5  -> coordinates (N,2) [x,y]

Output structure (under DATA_ROOT) — 3-way split, matching MoNuSeg/CoNIC:
  train/<id>.png  val/<id>.png  test/<id>.png        (symlinks)
  train_gt/<id>.txt  val_gt/<id>.txt  test_gt/<id>.txt  (merged pos+neg coords, "x y" per line)
  train.list  val.list  test.list                    (relative_img_path  relative_gt_path)

Mapping: images/train -> train.list, images/validation -> val.list,
images/test -> test.list. val.list is used for best.pth selection during
training (DATASETS.EVAL_LIST val.list); test.list is the final held-out set.
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
                    pts = np.asarray(f['coordinates'][:])  # [x, y]; may be (N,2), (2,) or empty
                    pts = pts.reshape(-1, 2)               # normalize: handles single-point / empty h5
                    if len(pts):
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

    # 3-way split (train / val / test), consistent with MoNuSeg & CoNIC
    convert_split(root,
                  split_name='train',
                  out_split='train',
                  out_gt_dir=os.path.join(root, 'train_gt'),
                  list_name='train.list')

    convert_split(root,
                  split_name='validation',
                  out_split='val',
                  out_gt_dir=os.path.join(root, 'val_gt'),
                  list_name='val.list')

    convert_split(root,
                  split_name='test',
                  out_split='test',
                  out_gt_dir=os.path.join(root, 'test_gt'),
                  list_name='test.list')

    print('Done. Data root:', root)


if __name__ == '__main__':
    main()
