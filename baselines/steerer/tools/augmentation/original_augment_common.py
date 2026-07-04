#!/usr/bin/env python3
"""Geometry helpers for original-format pathology augmentations."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True)
class Transform:
    scale: float
    affine: tuple[float, float, float, float, float, float]
    crop_x: int
    crop_y: int
    patch_size: int
    hflip: bool
    vflip: bool


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def pad_to_min_size(image: Image.Image, min_size: int, pad_value: int) -> Image.Image:
    width, height = image.size
    pad_w = max(0, min_size - width)
    pad_h = max(0, min_size - height)
    if pad_w == 0 and pad_h == 0:
        return image
    return ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=(pad_value, pad_value, pad_value))


def identity_affine() -> tuple[float, float, float, float, float, float]:
    return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def apply_affine_to_xy(x: float, y: float, affine: tuple[float, float, float, float, float, float]) -> tuple[float, float]:
    a, b, c, d, e, f = affine
    return a * x + b * y + c, d * x + e * y + f


def invert_affine(affine: tuple[float, float, float, float, float, float]) -> tuple[float, float, float, float, float, float]:
    a, b, c, d, e, f = affine
    det = a * e - b * d
    if abs(det) < 1e-8:
        return identity_affine()
    inv_a = e / det
    inv_b = -b / det
    inv_d = -d / det
    inv_e = a / det
    inv_c = -(inv_a * c + inv_b * f)
    inv_f = -(inv_d * c + inv_e * f)
    return inv_a, inv_b, inv_c, inv_d, inv_e, inv_f


def sample_affine(
    width: int,
    height: int,
    *,
    rng: random.Random,
    affine_enabled: bool,
    affine_prob: float,
    rotate_deg: float,
    translate_frac: float,
    shear_deg: float,
    affine_scale_min: float,
    affine_scale_max: float,
) -> tuple[float, float, float, float, float, float]:
    if not affine_enabled or rng.random() >= affine_prob:
        return identity_affine()

    angle = math.radians(rng.uniform(-rotate_deg, rotate_deg))
    shear = math.radians(rng.uniform(-shear_deg, shear_deg))
    affine_scale = rng.uniform(affine_scale_min, affine_scale_max)
    tx = rng.uniform(-translate_frac, translate_frac) * width
    ty = rng.uniform(-translate_frac, translate_frac) * height

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    shear_tan = math.tan(shear)

    # Forward matrix: scale -> x-shear -> rotation, around image center.
    m00 = affine_scale * (cos_a - sin_a * shear_tan)
    m01 = -affine_scale * sin_a
    m10 = affine_scale * (sin_a + cos_a * shear_tan)
    m11 = affine_scale * cos_a

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    c = cx + tx - (m00 * cx + m01 * cy)
    f = cy + ty - (m10 * cx + m11 * cy)
    return m00, m01, c, m10, m11, f


def apply_image_affine(
    image: Image.Image,
    affine: tuple[float, float, float, float, float, float],
    pad_value: int,
) -> Image.Image:
    if affine == identity_affine():
        return image
    inverse = invert_affine(affine)
    return image.transform(
        image.size,
        Image.AFFINE,
        inverse,
        resample=Image.BILINEAR,
        fillcolor=(pad_value, pad_value, pad_value),
    )


def sample_image_transform(
    image: Image.Image,
    *,
    rng: random.Random,
    patch_size: int,
    scale_min: float,
    scale_max: float,
    hflip_prob: float,
    vflip_prob: float,
    pad_value: int,
    affine_enabled: bool = True,
    affine_prob: float = 1.0,
    rotate_deg: float = 179.0,
    translate_frac: float = 0.01,
    shear_deg: float = 5.0,
    affine_scale_min: float = 0.8,
    affine_scale_max: float = 1.2,
) -> tuple[Image.Image, Transform]:
    scale = rng.uniform(scale_min, scale_max)
    scaled_w = max(1, int(round(image.width * scale)))
    scaled_h = max(1, int(round(image.height * scale)))
    aug_image = image.resize((scaled_w, scaled_h), Image.BILINEAR)
    affine = sample_affine(
        scaled_w,
        scaled_h,
        rng=rng,
        affine_enabled=affine_enabled,
        affine_prob=affine_prob,
        rotate_deg=rotate_deg,
        translate_frac=translate_frac,
        shear_deg=shear_deg,
        affine_scale_min=affine_scale_min,
        affine_scale_max=affine_scale_max,
    )
    aug_image = apply_image_affine(aug_image, affine, pad_value)
    aug_image = pad_to_min_size(aug_image, patch_size, pad_value)

    width, height = aug_image.size
    crop_x = rng.randint(0, width - patch_size)
    crop_y = rng.randint(0, height - patch_size)
    patch = aug_image.crop((crop_x, crop_y, crop_x + patch_size, crop_y + patch_size))

    hflip = rng.random() < hflip_prob
    if hflip:
        patch = ImageOps.mirror(patch)

    vflip = rng.random() < vflip_prob
    if vflip:
        patch = ImageOps.flip(patch)

    return patch, Transform(
        scale=scale,
        affine=affine,
        crop_x=crop_x,
        crop_y=crop_y,
        patch_size=patch_size,
        hflip=hflip,
        vflip=vflip,
    )


def transform_point(point: list[float] | tuple[float, float], transform: Transform) -> list[float] | None:
    x = float(point[0]) * transform.scale
    y = float(point[1]) * transform.scale
    x, y = apply_affine_to_xy(x, y, transform.affine)
    x -= transform.crop_x
    y -= transform.crop_y
    if not (0 <= x < transform.patch_size and 0 <= y < transform.patch_size):
        return None
    max_xy = transform.patch_size - 1
    if transform.hflip:
        x = max_xy - x
    if transform.vflip:
        y = max_xy - y
    return [x, y]


def transform_points(points: list[list[float]], transform: Transform) -> list[list[float]]:
    transformed: list[list[float]] = []
    for point in points:
        new_point = transform_point(point, transform)
        if new_point is not None:
            transformed.append(new_point)
    return transformed


def polygon_area(vertices: list[tuple[float, float]]) -> float:
    if len(vertices) < 3:
        return 0.0
    total = 0.0
    for idx, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(idx + 1) % len(vertices)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def clip_polygon(vertices: list[tuple[float, float]], max_xy: float) -> list[tuple[float, float]]:
    def clip_edge(poly, inside, intersect):
        if not poly:
            return []
        output = []
        prev = poly[-1]
        prev_inside = inside(prev)
        for curr in poly:
            curr_inside = inside(curr)
            if curr_inside:
                if not prev_inside:
                    output.append(intersect(prev, curr))
                output.append(curr)
            elif prev_inside:
                output.append(intersect(prev, curr))
            prev = curr
            prev_inside = curr_inside
        return output

    def intersect_x(p1, p2, x_bound):
        x1, y1 = p1
        x2, y2 = p2
        if x2 == x1:
            return x_bound, y1
        t = (x_bound - x1) / (x2 - x1)
        return x_bound, y1 + t * (y2 - y1)

    def intersect_y(p1, p2, y_bound):
        x1, y1 = p1
        x2, y2 = p2
        if y2 == y1:
            return x1, y_bound
        t = (y_bound - y1) / (y2 - y1)
        return x1 + t * (x2 - x1), y_bound

    poly = vertices
    poly = clip_edge(poly, lambda p: p[0] >= 0.0, lambda a, b: intersect_x(a, b, 0.0))
    poly = clip_edge(poly, lambda p: p[0] <= max_xy, lambda a, b: intersect_x(a, b, max_xy))
    poly = clip_edge(poly, lambda p: p[1] >= 0.0, lambda a, b: intersect_y(a, b, 0.0))
    poly = clip_edge(poly, lambda p: p[1] <= max_xy, lambda a, b: intersect_y(a, b, max_xy))
    return poly


def transform_polygon(vertices: list[tuple[float, float]], transform: Transform) -> list[tuple[float, float]]:
    shifted = []
    for x, y in vertices:
        ax, ay = apply_affine_to_xy(x * transform.scale, y * transform.scale, transform.affine)
        shifted.append((ax - transform.crop_x, ay - transform.crop_y))
    max_xy = float(transform.patch_size - 1)
    clipped = clip_polygon(shifted, max_xy)
    if transform.hflip:
        clipped = [(max_xy - x, y) for x, y in clipped]
    if transform.vflip:
        clipped = [(x, max_xy - y) for x, y in clipped]
    return clipped


def adjust_hsv(image: Image.Image, hue_deg: float, saturation_delta: float) -> Image.Image:
    hsv = np.asarray(image.convert("HSV"), dtype=np.int16)
    hue_delta = int(round(hue_deg / 360.0 * 255.0))
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 256
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + int(round(saturation_delta * 255.0)), 0, 255)
    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")


def apply_union_pixel_augment(
    image: Image.Image,
    *,
    rng: random.Random,
    pixel_aug_enabled: bool = True,
    blur_noise_prob: float = 1.0,
    color_aug_prob: float = 1.0,
) -> Image.Image:
    if not pixel_aug_enabled:
        return image

    output = image

    if rng.random() < blur_noise_prob:
        choice = rng.choice(("gaussian_blur", "median_blur", "gaussian_noise"))
        if choice == "gaussian_blur":
            kernel = rng.choice((1, 3, 5))
            if kernel > 1:
                output = output.filter(ImageFilter.GaussianBlur(radius=(kernel - 1) / 2.0))
        elif choice == "median_blur":
            kernel = rng.choice((1, 3, 5))
            if kernel > 1:
                output = output.filter(ImageFilter.MedianFilter(size=kernel))
        else:
            arr = np.asarray(output).astype(np.float32)
            sigma = rng.uniform(0.0, 12.75)
            noise = np.random.default_rng(rng.randrange(2**32)).normal(0.0, sigma, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            output = Image.fromarray(arr, mode="RGB")

    if rng.random() < color_aug_prob:
        output = adjust_hsv(
            output,
            hue_deg=rng.uniform(-8.0, 8.0),
            saturation_delta=rng.uniform(-0.2, 0.2),
        )
        arr = np.asarray(output).astype(np.float32)
        arr = np.clip(arr + rng.uniform(-26.0, 26.0), 0, 255).astype(np.uint8)
        output = Image.fromarray(arr, mode="RGB")
        output = ImageEnhance.Contrast(output).enhance(rng.uniform(0.75, 1.25))

    return output
