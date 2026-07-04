"""
Convert CoNIC dataset (npy format) to APGCC txt format.

CoNIC structure:
  data/images.npy   uint8  (4981, 256, 256, 3)
  data/labels.npy   uint16 (4981, 256, 256, 2)
                    channel 0: instance map  (0=bg, 1..N=cells)
                    channel 1: class map     (0=bg, 1..6=cell types)
  data/patch_info.csv  source image name for each patch (used for split)

Output (under DATA_ROOT):
  images/train/<id>.png
  images/val/<id>.png
  train_gt/<id>.txt     centroid per cell, one "x y" per line
  val_gt/<id>.txt
  train/                symlinks -> images/train/
  val/                  symlinks -> images/val/
  train.list
  test.list
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import cv2


def extract_centroids(inst_map):
    ids = np.unique(inst_map)
    ids = ids[ids > 0]
    centroids = []
    for uid in ids:
        ys, xs = np.where(inst_map == uid)
        centroids.append((float(xs.mean()), float(ys.mean())))
    return centroids


def save_split(data_root, indices, images, labels, out_name):
    img_out_dir = os.path.join(data_root, 'images', out_name)
    gt_out_dir  = os.path.join(data_root, f'{out_name}_gt')
    link_dir    = os.path.join(data_root, out_name)
    for d in [img_out_dir, gt_out_dir, link_dir]:
        os.makedirs(d, exist_ok=True)

    list_lines = []
    for idx in indices:
        img_fname = f'{idx}.png'
        img_path  = os.path.join(img_out_dir, img_fname)
        cv2.imwrite(img_path, cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR))

        centroids = extract_centroids(labels[idx, :, :, 0])
        gt_path = os.path.join(gt_out_dir, f'{idx}.txt')
        with open(gt_path, 'w') as f:
            for cx, cy in centroids:
                f.write(f'{cx:.2f} {cy:.2f}\n')

        link_path = os.path.join(link_dir, img_fname)
        if not os.path.exists(link_path):
            os.symlink(img_path, link_path)

        rel_img = os.path.join(out_name, img_fname)
        rel_gt  = os.path.relpath(gt_path, data_root)
        list_lines.append(f'{rel_img} {rel_gt}')

    list_file = 'train.list' if out_name == 'train' else 'test.list'
    with open(os.path.join(data_root, list_file), 'w') as f:
        f.write('\n'.join(list_lines) + '\n')

    print(f'[{out_name}] {len(list_lines)} patches -> {list_file}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', help='path to CoNIC root, e.g. /mnt/data1/llx/CoNICdata')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    root = args.data_root
    print('Loading npy files...')
    images = np.load(os.path.join(root, 'data', 'images.npy'))
    labels = np.load(os.path.join(root, 'data', 'labels.npy'))
    info   = pd.read_csv(os.path.join(root, 'data', 'patch_info.csv'))

    # split by source image to avoid leakage
    info['src'] = info['patch_info'].str.rsplit('-', n=1).str[0]
    src_images = sorted(info['src'].unique())
    random.seed(args.seed)
    random.shuffle(src_images)
    n_val = max(1, int(len(src_images) * args.val_ratio))
    val_srcs   = set(src_images[:n_val])
    train_srcs = set(src_images[n_val:])

    train_idx = info.index[info['src'].isin(train_srcs)].tolist()
    val_idx   = info.index[info['src'].isin(val_srcs)].tolist()
    print(f'Source images: total={len(src_images)}  train={len(train_srcs)}  val={len(val_srcs)}')
    print(f'Patches:       total={len(info)}  train={len(train_idx)}  val={len(val_idx)}')

    save_split(root, train_idx, images, labels, 'train')
    save_split(root, val_idx,   images, labels, 'val')
    print('Done.')


if __name__ == '__main__':
    main()
