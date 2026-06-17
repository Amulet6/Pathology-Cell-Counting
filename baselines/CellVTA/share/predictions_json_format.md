# predictions.json 统一格式（草案 v1.0）

> 待队友确认后冻结。本文档发给所有人对照生成，统一评估脚本 `centroid_eval.py` 基于此格式。

## 一、JSON 结构

```json
{
  "metadata": {
    "dataset": "MoNuSeg",
    "method": "HoVer-Net",
    "role": "gt",
    "extraction_method": "polygon_area_weighted_centroid",
    "coordinate_order": "xy",
    "coordinate_unit": "pixel",
    "matching_thresholds_px": [6, 12, 24]
  },
  "samples": [
    {
      "id": "TCGA-2Z-A9J9-01A-01-TS1",
      "points": [[x1, y1], [x2, y2], ...]
    }
  ]
}
```

## 二、metadata 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `dataset` | string | `"MoNuSeg"` 或 `"CoNIC"` |
| `method` | string | 方法名（`"HoVer-Net"`, `"CellViT"`, ...）。GT 文件填 `"ground_truth"` |
| `role` | string | `"gt"` 或 `"pred"` |
| `extraction_method` | string | GT: `"polygon_area_weighted_centroid"` (MoNuSeg) 或 `"pixel_mean_of_instance_mask"` (CoNIC)；Pred: 各方法自述提取方式 |
| `coordinate_order` | string | 固定 `"xy"`——即 `[x, y]` |
| `coordinate_unit` | string | 固定 `"pixel"` |
| `matching_thresholds_px` | [int] | 评估用距离阈值列表 `[6, 12, 24]` |

## 三、samples 字段

每个样本是一个对象：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 样本唯一标识。MoNuSeg 用 TCGA barcode；CoNIC 用 `conic_<patch_idx>` 格式（从 `split_info.csv` 的 `patch_idx` 取值，如 `conic_00055`） |
| `points` | [[float, float]] | 点坐标列表 `[[x1,y1], [x2,y2], ...]`。空图传 `[]` |

## 四、GT 生成方式（所有人统一）

| 数据集 | 脚本 | 命令 |
|------|------|------|
| MoNuSeg | `label_to_centroids(1).py monuseg` | `python label_to_centroids.py monuseg --xml_root <path> --output <out>` |
| CoNIC | `label_to_centroids(1).py conic` | `python label_to_centroids.py conic --labels_npy <path> --output <out>` |

之后需按 split 过滤出 test 样本（参见第六节）。

## 五、评估脚本

`centroid_eval.py`——读取 GT 和 Pred 两个 `predictions.json`，输出完整评估结果（MAE/MSE/RMSE + Precision/Recall/F1 @ 6/12/24px）。

```bash
python centroid_eval.py --gt gt.json --pred pred.json --thresholds 6 12 24
```

输出 `*_centroid_eval.json`：

```json
{
  "counting": {"mae": ..., "mse": ..., "rmse": ..., "total_gt": ..., "total_pred": ..., "total_err_pct": ...},
  "localization": {
    "12px": {"precision": ..., "recall": ..., "f1": ..., "tp": ..., "fp": ..., "fn": ...}
  }
}
```

## 六、待确认事项

1. **`id` 命名规范**：MoNuSeg 用 TCGA barcode、CoNIC 用 `conic_<patch_idx>` 是否 OK？GT 和 Pred 的 id 需要一一对应，否则评估脚本无法匹配
2. **`extraction_method` 值**：是自由文本还是统一枚举（如 `"cv2_moments"`, `"np_mean"`, `"model_output"`）？
3. **空图表示**：`"points": []` 是否接受？（CoNIC 有 140 张零细胞 patch）
4. **阈值列表**：`[6, 12, 24]` 是否 OK？
