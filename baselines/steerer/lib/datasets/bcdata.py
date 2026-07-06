"""Shared pathology dataset adapter.

The upstream NWPU loader assumes numeric JPEG filenames. Converted pathology
datasets preserve PNG/TIFF suffixes, so this module changes file discovery
while reusing the official STEERER training and augmentation pipeline.
"""


import json
import os

from .nwpu import NWPU


class BCData(NWPU):
    """Load converted pathology samples with flexible image suffixes."""

    # Ordered fallback keeps legacy datasets without JSON img_id deterministic.
    image_suffixes = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    def read_files(self):
        """Build relative-path records from the active split list."""
        files = []
        for item in self.img_list:
            image_id = item[0]
            label_path = os.path.join('jsons', image_id + '.json')
            image_path = self._resolve_image_path(image_id, label_path)
            record = {
                "img": image_path,
                "label": label_path,
                "name": image_id,
            }
            if 'test' not in self.list_path:
                record["weight"] = 1
            files.append(record)
        return files

    def _resolve_image_path(self, image_id, label_path):
        """Prefer JSON metadata, then scan suffixes; fail on missing samples."""
        label_abs = os.path.join(self.root, label_path)
        if os.path.isfile(label_abs):
            with open(label_abs, 'r') as f:
                info = json.load(f)
            img_id = info.get('img_id')
            if img_id:
                image_path = os.path.join('images', img_id)
                if os.path.isfile(os.path.join(self.root, image_path)):
                    return image_path

        for suffix in self.image_suffixes:
            image_path = os.path.join('images', image_id + suffix)
            if os.path.isfile(os.path.join(self.root, image_path)):
                return image_path

        raise FileNotFoundError(
            'Could not find BCData image for id {} under {}'.format(
                image_id, os.path.join(self.root, 'images')))
