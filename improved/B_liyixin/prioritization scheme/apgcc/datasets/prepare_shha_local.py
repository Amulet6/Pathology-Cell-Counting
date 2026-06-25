# Convert original ShanghaiTech part_A_final layout -> APGCC list/txt format.
# Layout assumed:
#   {SRC}/train_data/images/IMG_*.jpg  + ground_truth/GT_IMG_*.mat
#   {SRC}/test_data/images/IMG_*.jpg   + ground_truth/GT_IMG_*.mat
# Output:
#   {OUT}/labels/{train,test}/IMG_*.txt   (one "x y" per line)
#   {OUT}/train.list, {OUT}/test.list     (absolute "image_path label_path")
# Usage: python prepare_shha_local.py <SRC> <OUT>
import os
import sys
from scipy.io import loadmat


def get_points(mat_path):
    m = loadmat(mat_path)
    return m['image_info'][0][0][0][0][0]


def convert(src, out):
    splits = {'train': 'train_data', 'test': 'test_data'}
    for split, sub in splits.items():
        img_dir = os.path.join(src, sub, 'images')
        gt_dir = os.path.join(src, sub, 'ground_truth')
        lbl_dir = os.path.join(out, 'labels', split)
        os.makedirs(lbl_dir, exist_ok=True)

        images = sorted(f for f in os.listdir(img_dir) if f.lower().endswith('.jpg'))
        list_lines = []
        for img in images:
            img_path = os.path.join(img_dir, img)
            gt_path = os.path.join(gt_dir, 'GT_' + img.replace('.jpg', '.mat'))
            pts = get_points(gt_path)
            lbl_path = os.path.join(lbl_dir, img.replace('.jpg', '.txt'))
            with open(lbl_path, 'w') as fp:
                for p in pts:
                    fp.write('{} {}\n'.format(p[0], p[1]))
            list_lines.append('{} {}'.format(img_path, lbl_path))

        with open(os.path.join(out, '{}.list'.format(split)), 'w') as fp:
            fp.write('\n'.join(list_lines) + '\n')
        print('[{}] {} images -> {}.list'.format(split, len(images), split))


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else '/data1/llx/part_A_final'
    out = sys.argv[2] if len(sys.argv) > 2 else '/data1/llx/part_A_apgcc'
    convert(src, out)
    print('Done. Set DATASETS.DATA_ROOT to:', out)
