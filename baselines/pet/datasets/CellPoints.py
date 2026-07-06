import csv
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


class CellPoints(Dataset):
    def __init__(self, data_root, transform=None, train=False, flip=False, patch_size=256,
                 split=None, aug_mode='native', dataset_name='CELL'):
        self.root_path = Path(data_root)
        self.split = split or ('train' if train else 'val')
        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = patch_size
        self.aug_mode = aug_mode
        self.dataset_name = dataset_name
        self.samples = self._load_samples()
        max_samples = getattr(self, 'max_samples', None)
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.nSamples = len(self.samples)
        if self.nSamples == 0:
            raise RuntimeError(f'No samples found under {self.root_path / self.split}')

    def _load_samples(self):
        manifest = self.root_path / f'{self.split}.csv'
        if manifest.exists():
            return self._load_manifest(manifest)

        img_dir = self.root_path / self.split / 'images'
        pts_dir = self.root_path / self.split / 'points'
        if not img_dir.exists():
            return []

        samples = []
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            for suffix in ('.npy', '.json', '.csv', '.txt'):
                points_path = pts_dir / f'{img_path.stem}{suffix}'
                if points_path.exists():
                    samples.append((str(img_path), str(points_path)))
                    break
        return samples

    def _load_manifest(self, manifest):
        samples = []
        with open(manifest, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = Path(row['image'])
                points_path = Path(row['points'])
                if not img_path.is_absolute():
                    img_path = self.root_path / img_path
                if not points_path.is_absolute():
                    points_path = self.root_path / points_path
                samples.append((str(img_path), str(points_path)))
        return samples

    def compute_density(self, points):
        if points.shape[0] == 0:
            return torch.tensor(999.0).reshape(-1)
        points_tensor = torch.from_numpy(points.copy()).float()
        dist = torch.cdist(points_tensor, points_tensor, p=2)
        if points_tensor.shape[0] > 1:
            density = dist.sort(dim=1)[0][:, 1].mean().reshape(-1)
        else:
            density = torch.tensor(999.0).reshape(-1)
        return density

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img_path, points_path = self.samples[index]
        img, points = load_data(img_path, points_path)
        points = points.astype(float)

        if self.train and self.aug_mode == 'unified':
            img, points = unified_augment(img, points, self.patch_size, self.dataset_name)
            if self.transform is not None:
                img = self.transform(img)
            img = torch.Tensor(img)
        else:
            if self.transform is not None:
                img = self.transform(img)
            img = torch.Tensor(img)

            if self.train:
                scale_range = [0.8, 1.2]
                min_size = min(img.shape[1:])
                scale = random.uniform(*scale_range)
                if scale * min_size > self.patch_size:
                    img = torch.nn.functional.interpolate(
                        img.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False
                    ).squeeze(0)
                    points *= scale

                img, points = random_crop(img, points, patch_size=self.patch_size)

            if random.random() > 0.5 and self.train and self.flip:
                img = torch.flip(img, dims=[2])
                points[:, 1] = self.patch_size - points[:, 1]

        target = {
            'points': torch.Tensor(points),
            'labels': torch.ones([points.shape[0]]).long(),
        }
        if self.train:
            target['density'] = self.compute_density(points)
        else:
            target['image_path'] = img_path
        return img, target


def load_data(img_path, points_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f'Failed to read image: {img_path}')
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    points = load_points(points_path)
    return img, points


def load_points(points_path):
    points_path = Path(points_path)
    suffix = points_path.suffix.lower()
    if suffix == '.npy':
        points = np.load(points_path)
    elif suffix == '.json':
        with open(points_path, encoding='utf-8') as f:
            data = json.load(f)
        points = data['points'] if isinstance(data, dict) else data
    elif suffix in {'.csv', '.txt'}:
        points = np.loadtxt(points_path, delimiter=',' if suffix == '.csv' else None)
    else:
        raise ValueError(f'Unsupported point annotation: {points_path}')

    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return points


def has_points(points_path):
    try:
        return load_points(points_path).shape[0] > 0
    except Exception:
        return False


def random_crop(img, points, patch_size=256):
    patch_h = patch_size
    patch_w = patch_size

    start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
    start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h = start_h + patch_h
    end_w = start_w + patch_w
    idx = (
        (points[:, 0] >= start_h)
        & (points[:, 0] <= end_h)
        & (points[:, 1] >= start_w)
        & (points[:, 1] <= end_w)
    )

    result_img = img[:, start_h:end_h, start_w:end_w]
    result_points = points[idx].copy()
    result_points[:, 0] -= start_h
    result_points[:, 1] -= start_w

    imgH, imgW = result_img.shape[-2:]
    fH, fW = patch_h / imgH, patch_w / imgW
    result_img = torch.nn.functional.interpolate(
        result_img.unsqueeze(0), (patch_h, patch_w), mode='bilinear', align_corners=False
    ).squeeze(0)
    result_points[:, 0] *= fH
    result_points[:, 1] *= fW
    return result_img, result_points


def pil_to_rgb_array(img):
    return np.asarray(img.convert('RGB'))


def resize_array(img, points, scale):
    h, w = img.shape[:2]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    points = points.copy()
    points *= scale
    return img, points


def random_crop_array(img, points, patch_size=256):
    h, w = img.shape[:2]
    start_h = random.randint(0, h - patch_size) if h > patch_size else 0
    start_w = random.randint(0, w - patch_size) if w > patch_size else 0
    end_h = start_h + patch_size
    end_w = start_w + patch_size
    idx = (
        (points[:, 0] >= start_h)
        & (points[:, 0] <= end_h)
        & (points[:, 1] >= start_w)
        & (points[:, 1] <= end_w)
    )
    crop = img[start_h:min(end_h, h), start_w:min(end_w, w)]
    result_points = points[idx].copy()
    result_points[:, 0] -= start_h
    result_points[:, 1] -= start_w

    crop_h, crop_w = crop.shape[:2]
    if crop_h == 0 or crop_w == 0:
        crop = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        return crop, np.empty((0, 2), dtype=np.float32)

    if crop_h != patch_size or crop_w != patch_size:
        f_h, f_w = patch_size / crop_h, patch_size / crop_w
        crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        result_points[:, 0] *= f_h
        result_points[:, 1] *= f_w
    return crop, result_points


def horizontal_flip_array(img, points, patch_size):
    img = np.ascontiguousarray(img[:, ::-1])
    points = points.copy()
    points[:, 1] = patch_size - points[:, 1]
    return img, points


def vertical_flip_array(img, points, patch_size):
    img = np.ascontiguousarray(img[::-1, :])
    points = points.copy()
    points[:, 0] = patch_size - points[:, 0]
    return img, points


def affine_array(img, points, patch_size):
    angle = random.uniform(-179.0, 179.0)
    shear = random.uniform(-5.0, 5.0)
    tx = random.uniform(-0.01, 0.01) * patch_size
    ty = random.uniform(-0.01, 0.01) * patch_size

    center = patch_size / 2.0
    cos_a, sin_a = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    shear_t = np.tan(np.deg2rad(shear))

    t1 = np.array([[1, 0, -center], [0, 1, -center], [0, 0, 1]], dtype=np.float32)
    rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
    shr = np.array([[1, shear_t, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    t2 = np.array([[1, 0, center + tx], [0, 1, center + ty], [0, 0, 1]], dtype=np.float32)
    mat = (t2 @ shr @ rot @ t1)[:2]

    warped = cv2.warpAffine(
        img, mat, (patch_size, patch_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    if points.shape[0] == 0:
        return warped, points

    xy1 = np.stack([points[:, 1], points[:, 0], np.ones(points.shape[0])], axis=1)
    xy_new = xy1 @ mat.T
    new_points = np.stack([xy_new[:, 1], xy_new[:, 0]], axis=1)
    keep = (
        (new_points[:, 0] >= 0)
        & (new_points[:, 0] <= patch_size)
        & (new_points[:, 1] >= 0)
        & (new_points[:, 1] <= patch_size)
    )
    return warped, new_points[keep]


def blur_or_noise(img):
    choice = random.choice(['gaussian_blur', 'median_blur', 'gaussian_noise'])
    if choice == 'gaussian_blur':
        ksize = random.choice([1, 3, 5])
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if choice == 'median_blur':
        ksize = random.choice([1, 3, 5])
        return cv2.medianBlur(img, ksize)
    sigma = random.uniform(0.0, 12.75)
    noise = np.random.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def color_jitter(img):
    img_f = img.astype(np.float32)
    contrast = random.uniform(0.75, 1.25)
    brightness = random.uniform(-26.0, 26.0)
    img_f = np.clip(img_f * contrast + brightness, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img_f, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-8.0, 8.0)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + random.uniform(-0.2, 0.2)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def unified_augment(img, points, patch_size, dataset_name):
    img = pil_to_rgb_array(img)
    points = points.copy()

    scale = random.uniform(0.8, 1.2)
    img, points = resize_array(img, points, scale)
    img, points = random_crop_array(img, points, patch_size=patch_size)

    if random.random() < 0.5:
        img, points = horizontal_flip_array(img, points, patch_size)
    if random.random() < 0.5:
        img, points = vertical_flip_array(img, points, patch_size)

    if dataset_name.lower() != 'conic':
        img, points = affine_array(img, points, patch_size)

    img = blur_or_noise(img)
    img = color_jitter(img)
    return Image.fromarray(img), points.astype(np.float32)


def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])

    data_root = args.data_path
    patch_size = getattr(args, 'patch_size', 256)
    aug_mode = getattr(args, 'aug_mode', 'native')
    dataset_name = getattr(args, 'dataset_file', 'CELL')
    if image_set == 'train':
        dataset = CellPoints(data_root, train=True, transform=transform, flip=True, patch_size=patch_size,
                             split='train', aug_mode=aug_mode, dataset_name=dataset_name)
        if not getattr(args, 'keep_empty_train', 0):
            dataset.samples = [sample for sample in dataset.samples if has_points(sample[1])]
            dataset.nSamples = len(dataset.samples)
        max_samples = getattr(args, 'max_train_samples', 0)
        if max_samples:
            dataset.samples = dataset.samples[:max_samples]
            dataset.nSamples = len(dataset.samples)
        return dataset
    if image_set in {'val', 'test'}:
        dataset = CellPoints(data_root, train=False, transform=transform, patch_size=patch_size,
                             split=image_set, aug_mode=aug_mode, dataset_name=dataset_name)
        max_samples = getattr(args, 'max_val_samples', 0)
        if max_samples:
            dataset.samples = dataset.samples[:max_samples]
            dataset.nSamples = len(dataset.samples)
        return dataset
    raise ValueError(f'image_set {image_set} not supported')
