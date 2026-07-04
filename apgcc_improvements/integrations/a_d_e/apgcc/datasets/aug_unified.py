"""Team-unified ONLINE data augmentation for APGCC (Route B).

Implements the pipeline in baselines/apgcc/data_augmentation_protocol.md, fixed order:
    random scale -> random affine -> random crop -> h/v flip -> blur/noise -> color jitter

All ops operate on a uint8 RGB image (H,W,3) and an Nx2 point array ([x, y] float).
Geometric ops (scale/affine/crop/flip) transform the points in sync; pixel ops
(blur/noise/color) touch the image only. Normalization is NOT done here (the dataset
applies ToTensor+Normalize afterwards, per protocol).

Produces `crop_number` patches per image (matching APGCC's multi-crop sampler):
scale+affine are applied once on the full image, then each patch is cropped and gets
its own flip/blur/color.
"""
import random

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# geometric (image + points)
# --------------------------------------------------------------------------- #
def random_scale(img, pts, smin, smax):
    s = random.uniform(smin, smax)
    h, w = img.shape[:2]
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if len(pts):
        pts = pts.copy()
        pts[:, 0] *= nw / float(w)
        pts[:, 1] *= nh / float(h)
    return img, pts


def random_affine(img, pts, scale_rng, trans_frac, shear_deg, rot_deg):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    angle = random.uniform(-rot_deg, rot_deg)
    sc = random.uniform(*scale_rng)
    shear = np.deg2rad(random.uniform(-shear_deg, shear_deg))
    tx = random.uniform(-trans_frac, trans_frac) * w
    ty = random.uniform(-trans_frac, trans_frac) * h

    R = cv2.getRotationMatrix2D((cx, cy), angle, sc)        # rotation + uniform scale about center
    R3 = np.vstack([R, [0.0, 0.0, 1.0]])
    Sh = np.array([[1.0, np.tan(shear), -np.tan(shear) * cy],  # x-shear about center
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    A = Sh @ R3
    A[0, 2] += tx
    A[1, 2] += ty
    M = A[:2]

    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    if len(pts):
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts = (M @ np.hstack([pts.astype(np.float64), ones]).T).T
    return img, pts


def random_crop_one(img, pts, size):
    # pad (reflect) if image smaller than crop size
    h, w = img.shape[:2]
    ph, pw = max(0, size - h), max(0, size - w)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        h, w = img.shape[:2]
    y0 = random.randint(0, h - size)
    x0 = random.randint(0, w - size)
    crop = img[y0:y0 + size, x0:x0 + size]
    if len(pts):
        m = (pts[:, 0] >= x0) & (pts[:, 0] < x0 + size) & \
            (pts[:, 1] >= y0) & (pts[:, 1] < y0 + size)
        cpts = pts[m].copy()
        cpts[:, 0] -= x0
        cpts[:, 1] -= y0
    else:
        cpts = np.empty((0, 2), dtype=np.float64)
    return crop, cpts


def hflip(img, pts):
    img = np.ascontiguousarray(img[:, ::-1])
    if len(pts):
        pts = pts.copy()
        pts[:, 0] = img.shape[1] - 1 - pts[:, 0]
    return img, pts


def vflip(img, pts):
    img = np.ascontiguousarray(img[::-1, :])
    if len(pts):
        pts = pts.copy()
        pts[:, 1] = img.shape[0] - 1 - pts[:, 1]
    return img, pts


# --------------------------------------------------------------------------- #
# pixel (image only)
# --------------------------------------------------------------------------- #
def blur_noise(img):
    """Gaussian blur / median blur / Gaussian noise, each 1/3."""
    r = random.random()
    if r < 1.0 / 3:
        k = random.choice([1, 3, 5])
        if k > 1:
            img = cv2.GaussianBlur(img, (k, k), 0)
    elif r < 2.0 / 3:
        k = random.choice([1, 3, 5])
        if k > 1:
            img = cv2.medianBlur(img, k)
    else:
        sigma = random.uniform(0, 12.75)
        noise = np.random.randn(*img.shape) * sigma
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def color_jitter(img):
    """hue U(-8,8), sat *(1+U(-0.2,0.2)), brightness +U(-26,26), contrast *U(0.75,1.25)."""
    brightness = random.uniform(-26, 26)
    contrast = random.uniform(0.75, 1.25)
    out = np.clip(img.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = random.uniform(-8, 8)
    sat = random.uniform(-0.2, 0.2)
    hsv[..., 0] = (hsv[..., 0] + hue) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sat), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# --------------------------------------------------------------------------- #
class UnifiedAug:
    """Callable: (img_uint8 HxWx3 RGB, pts Nx2 [x,y]) -> (list[patch_uint8], list[pts])."""

    def __init__(self, crop_size, crop_number, affine_on,
                 scale_rng=(0.8, 1.2), affine_scale=(0.8, 1.2), trans_frac=0.01,
                 shear_deg=5.0, rot_deg=179.0,
                 hflip_p=0.5, vflip_p=0.5, blur_noise_p=1.0, color_p=1.0):
        self.crop_size = crop_size
        self.crop_number = crop_number
        self.affine_on = affine_on
        self.scale_rng = scale_rng
        self.affine_scale = affine_scale
        self.trans_frac = trans_frac
        self.shear_deg = shear_deg
        self.rot_deg = rot_deg
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.blur_noise_p = blur_noise_p
        self.color_p = color_p

    def __call__(self, img, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        img, pts = random_scale(img, pts, *self.scale_rng)
        if self.affine_on:
            img, pts = random_affine(img, pts, self.affine_scale,
                                     self.trans_frac, self.shear_deg, self.rot_deg)
        patches, ppoints = [], []
        for _ in range(self.crop_number):
            cimg, cpts = random_crop_one(img, pts, self.crop_size)
            if random.random() < self.hflip_p:
                cimg, cpts = hflip(cimg, cpts)
            if random.random() < self.vflip_p:
                cimg, cpts = vflip(cimg, cpts)
            # safety: drop any point pushed to/over the border by sub-pixel flip rounding
            if len(cpts):
                s = self.crop_size
                m = (cpts[:, 0] >= 0) & (cpts[:, 0] < s) & (cpts[:, 1] >= 0) & (cpts[:, 1] < s)
                cpts = cpts[m]
            if random.random() < self.blur_noise_p:
                cimg = blur_noise(cimg)
            if random.random() < self.color_p:
                cimg = color_jitter(cimg)
            patches.append(np.ascontiguousarray(cimg))
            ppoints.append(cpts)
        return patches, ppoints
