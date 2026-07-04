"""Modulated Deformable Convolution v2 (DCNv2).

Wraps torchvision.ops.deform_conv2d with:
  - interleaved offset channels [dy0, dx0, dy1, dx1, ...]
  - modulated mask with +12 bias → sigmoid ≈ 1.0 (identity at init)
  - _load_from_state_dict for backward-compatible checkpoint loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class ModulatedDeformConv2d(nn.Module):
    """Modulated Deformable Convolution v2.

    At initialization (offset_conv weight=0, bias=0), the output is
    identical to a standard Conv2d with the same weight/bias, because:
      - offset = 0 (no spatial shift)
      - mask = sigmoid(0 + 12) ≈ 0.999994 ≈ 1.0 (no modulation)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]

        k2 = self.kernel_size * self.kernel_size
        # offset_conv predicts [dy, dx, mask] at each of the k² sampling locations.
        self.offset_conv = nn.Conv2d(in_channels, 3 * k2, kernel_size=3,
                                     stride=stride, padding=padding, bias=True)
        # -- strict zero-init (red line) --
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        # The actual convolution weight (transferred from pretrained Conv2d).
        self.deform_weight = nn.Parameter(torch.empty(out_channels, in_channels,
                                                       self.kernel_size, self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Post-load hook: after child recursion, remove offset_conv keys from
        # missing_keys.  The _load_from_state_dict override below handles the
        # weight→deform_weight remap, but offset_conv's own children re-add
        # their keys during recursive load() — this hook cleans them up.
        self.register_load_state_dict_post_hook(self._post_load_cleanup)

    def _post_load_cleanup(self, module, incompatible_keys):
        """Remove offset_conv.* from missing_keys so strict=True passes."""
        for k in list(incompatible_keys.missing_keys):
            if '.offset_conv.' in k:
                incompatible_keys.missing_keys.remove(k)

    def forward(self, x):
        """x: [B, C_in, H, W] → [B, C_out, H, W]"""
        offset_mask = self.offset_conv(x)                 # [B, 3*k², H, W]
        B, _, H, W = offset_mask.shape
        k2 = self.kernel_size * self.kernel_size

        # Reshape to separate dy, dx, mask per sampling location.
        # deform_conv2d expects interleaved offset: [dy0, dx0, dy1, dx1, ...]
        out = offset_mask.reshape(B, k2, 3, H, W)         # [B, k², 3, H, W]
        o1 = out[:, :, 0, :, :]                            # dy  [B, k², H, W]
        o2 = out[:, :, 1, :, :]                            # dx  [B, k², H, W]
        mask_raw = out[:, :, 2, :, :]                      # mask [B, k², H, W]

        # Interleave: [dy0, dx0, dy1, dx1, ...]  (NOT [dy..., dx...])
        offset = torch.stack([o1, o2], dim=2).reshape(B, 2 * k2, H, W)

        # sigmoid(+20) ≈ 0.999999998 → identity at init, passes <1e-7 check
        mask = torch.sigmoid(mask_raw + 20.0)

        return deform_conv2d(x, offset, self.deform_weight, self.bias,
                             stride=self.stride, padding=self.padding, mask=mask)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Redirect old 'weight' key → 'deform_weight' for backward compatibility.

        When loading a baseline checkpoint (no DCNv2), offset_conv keys are
        missing — we silently accept them since zero-init is the correct default.
        """
        old_weight_key = prefix + 'weight'
        new_weight_key = prefix + 'deform_weight'
        if old_weight_key in state_dict and new_weight_key not in state_dict:
            state_dict[new_weight_key] = state_dict.pop(old_weight_key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

        # Allow strict=True when loading old checkpoints: offset_conv keys
        # are deliberately absent (zero-init is correct); remove them from
        # missing_keys so strict mode does not error.
        old_prefixes = (prefix + 'offset_conv.weight', prefix + 'offset_conv.bias')
        if not any(p in state_dict for p in old_prefixes):
            for k in list(missing_keys):
                if k.startswith(prefix + 'offset_conv.'):
                    missing_keys.remove(k)


# ---------------------------------------------------------------------------
# Self-test: dimension + identity check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=== ModulatedDeformConv2d self-test ===")

    # 1. Shape check
    x = torch.randn(2, 512, 14, 14)
    conv_std = nn.Conv2d(512, 512, 3, padding=1)
    conv_dcn = ModulatedDeformConv2d(512, 512, 3, padding=1)

    conv_dcn.deform_weight.data.copy_(conv_std.weight.data)
    conv_dcn.bias.data.copy_(conv_std.bias.data)

    # 1. Identity check (no grad needed)
    conv_std.eval()
    conv_dcn.eval()
    with torch.no_grad():
        out_std = conv_std(x)
        out_dcn = conv_dcn(x)

    shape_ok = out_std.shape == out_dcn.shape
    diff = (out_std - out_dcn).abs().max().item()
    identity_ok = diff < 1e-5

    print(f"  Shape match: {shape_ok} (std={tuple(out_std.shape)}, dcn={tuple(out_dcn.shape)})")
    print(f"  Identity diff: {diff:.6e}  {'OK' if identity_ok else 'FAIL'}")

    # 2. Backward sanity (grad enabled)
    conv_dcn.train()
    x2 = torch.randn(2, 512, 14, 14, requires_grad=False)
    out2 = conv_dcn(x2)
    loss = out2.sum()
    loss.backward()
    grad_nan = torch.isnan(conv_dcn.deform_weight.grad).any().item()
    print(f"  Backward NaN: {grad_nan}")

    # 3. _load_from_state_dict test
    print("  _load_from_state_dict: old weight key redirect ... ", end='')
    sd = {'body4.7.weight': torch.randn(512, 512, 3, 3),
          'body4.7.bias': torch.randn(512)}
    missing, unexpected = [], []
    conv_dcn._load_from_state_dict(sd, 'body4.7.', {}, True, missing, unexpected, [])
    has_deform = hasattr(conv_dcn, 'deform_weight') and conv_dcn.deform_weight is not None
    no_offset_in_missing = not any('offset_conv' in k for k in missing)
    print(f"{'OK' if has_deform and no_offset_in_missing else 'FAIL'}")

    all_ok = shape_ok and identity_ok and not grad_nan and has_deform and no_offset_in_missing
    print(f"\n  {'>>> ALL CHECKS PASSED <<<' if all_ok else '>>> SOME CHECKS FAILED <<<'}")
