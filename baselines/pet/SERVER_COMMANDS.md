# Server Run Commands

## MoNuSeg

确认服务器上原始数据路径：

```bash
find /root/autodl-tmp/data_raw -maxdepth 3 -type d -name "MoNuSeg"
find /root/autodl-tmp/data_raw/MoNuSeg -name "*.xml" | head
```

转换为 PET 点标注格式：

```bash
cd /path/to/Pathology-Cell-Counting
python data_conversion/convert_to_points.py --dataset monuseg --src_root /root/autodl-tmp/data_raw/MoNuSeg --out_root data/MoNuSeg_pet --val_ratio 0.2
```

小样本测试：

```bash
export OMP_NUM_THREADS=4
python baselines/pet/main.py --dataset_file MoNuSeg --data_path data/MoNuSeg_pet --device cuda --num_workers 0 --batch_size 1 --epochs 2 --eval_freq 1 --output_dir monuseg_debug --max_train_samples 8 --max_val_samples 2
```

正式训练：

```bash
screen -S pet_monuseg
export OMP_NUM_THREADS=4
python baselines/pet/main.py --dataset_file MoNuSeg --data_path data/MoNuSeg_pet --device cuda --num_workers 2 --batch_size 4 --epochs 100 --eval_freq 10 --output_dir monuseg_100ep_bs4; shutdown -h now
```

进入后台后按：

```text
Ctrl + A
D
```

训练完成后评估：

```bash
python baselines/pet/eval.py --dataset_file MoNuSeg --data_path data/MoNuSeg_pet --device cuda --num_workers 2 --resume outputs/MoNuSeg/monuseg_100ep_bs4/best_checkpoint.pth --vis_dir outputs/MoNuSeg/monuseg_100ep_bs4/vis_best --loc_radius 24
```

查看日志：

```bash
tail -80 outputs/MoNuSeg/monuseg_100ep_bs4/run_log.txt
grep "best mae" outputs/MoNuSeg/monuseg_100ep_bs4/run_log.txt | tail
```
