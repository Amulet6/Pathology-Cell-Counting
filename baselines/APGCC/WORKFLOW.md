# APGCC 复现 · Git 工作流速查 (个人备忘)

> 仓库: https://github.com/Amulet6/Pathology-Cell-Counting
> 我的分支: `feature/apgcc-baseline`
> 我的工作目录: `baselines/APGCC/`
> 本地路径: `/home/lixinli/Pathology-Cell-Counting`

---

## 0. 一次性：仓库已 clone

仓库已经在 `/home/lixinli/Pathology-Cell-Counting`，**不用再 clone**。
（万一换电脑要重新拉：`git clone https://github.com/Amulet6/Pathology-Cell-Counting.git`）

进入仓库：

```bash
cd /home/lixinli/Pathology-Cell-Counting
```

---

## 1. 每天开工前：拉取队友的最新代码

```bash
# 先切到 main，拉最新
git checkout main
git pull origin main

# 切回我自己的分支
git checkout feature/apgcc-baseline

# 把 main 的最新进度合并到我的分支（保持不落后，减少冲突）
git merge main
```

> 如果 merge 出现冲突：编辑冲突文件 → `git add <文件>` → `git commit`。

---

## 2. 干活 → 提交 (commit)

```bash
# 看改了哪些文件
git status

# 看具体改动内容
git diff

# 添加要提交的文件（只加 apgcc 目录，别误加别人的）
git add baselines/APGCC

# 或者全部添加（注意别加进数据/权重，.gitignore 已挡掉 *.pth / data/ 等）
git add -A

# 提交，写清楚做了什么
git commit -m "feat(apgcc): BCData 点格式转换器"
```

**commit message 建议格式**（和队友风格一致）：
- `feat(apgcc): xxx` 新功能
- `fix(apgcc): xxx` 修 bug
- `docs(apgcc): xxx` 改文档
- `exp(apgcc): xxx` 跑实验/记录结果

---

## 3. 推送到 GitHub (push)

```bash
# 第一次 push 这个分支（建立追踪）
git push -u origin feature/apgcc-baseline

# 之后再 push 就直接
git push
```

---

## 4. 开 Pull Request (PR) 合并到 main

推送后，去 GitHub 仓库页面会提示 "Compare & pull request"，点它，base 选 `main`，
填标题说明，请队友 review 后合并。

命令行方式（需装 gh CLI）：

```bash
gh pr create --base main --head feature/apgcc-baseline \
  --title "APGCC baseline" --body "复现 APGCC 并迁移到病理细胞数据集"
```

PR 合并后，更新本地 main：

```bash
git checkout main
git pull origin main
```

---

## 5. 常用查看命令

```bash
git log --oneline -10          # 看最近提交
git branch -a                  # 看所有分支
git checkout -- <文件>          # 放弃某文件的未提交改动
git stash                      # 临时存起未提交改动；git stash pop 恢复
git check-ignore <文件>         # 确认某文件是否被 .gitignore 忽略
```

---

## ⚠️ 注意事项

1. **不要提交数据集和权重**：`data/`、`*.pth`、`output/` 已被 `.gitignore` 忽略，别强行 `git add -f`。
2. **只动自己的目录**：改动尽量限制在 `baselines/APGCC/`，避免和队友冲突。
3. **本基线直接改官方代码**（不分 official/+src/），和队友约定不同，push 前在群里说明一下。
4. **勤 commit、勤 pull**：每完成一小块就 commit；每天开工先 pull main 合并，减少冲突。


# 测试（训练完后，test() 固定用 test.list）
python main.py -t -c ./configs/MoNuSeg_finetune.yml \
  GPU_ID 2 \
  TEST.WEIGHT ./output/MoNuSeg_finetune/best.pth

  # 训练：覆盖 DATA_ROOT（配置里的 /mnt 路径不存在）+ EVAL_LIST 用 val.list 做 3-way
nohup python main.py -c ./configs/CoNIC_finetune.yml \
  GPU_ID 3 \
  DATASETS.DATA_ROOT /data1/llx/CoNICdata \
  DATASETS.EVAL_LIST val.list \
  > train_conic.log 2>&1 &

# 测试
python main.py -t -c ./configs/CoNIC_finetune.yml \
  GPU_ID 3 \
  DATASETS.DATA_ROOT /data1/llx/CoNICdata \
  TEST.WEIGHT ./output/CoNIC_finetune/best.pth

  # 训练：覆盖 DATA_ROOT(配置是 /mnt 错路径) + EVAL_LIST 用 val.list
nohup python main.py -c ./configs/BCData_finetune.yml \
  GPU_ID 5 \
  DATASETS.DATA_ROOT /data1/llx/BCData \
  DATASETS.EVAL_LIST val.list \
  SOLVER.LOG_FREQ 1 SOLVER.EVAL_FREQ 1 \
  > train_bcdata.log 2>&1 &

# 测试
python main.py -t -c ./configs/BCData_finetune.yml \
  GPU_ID 5 \
  DATASETS.DATA_ROOT /data1/llx/BCData \
  TEST.WEIGHT ./output/BCData_finetune/best.pth

  PY=/home/lixinli/anaconda3/envs/apgcc/bin/python
# 例：CoNIC（其余替换 config/weight/data-root）
$PY eval_centroid.py --config ./configs/CoNIC_unified.yml \
  --weight ./output/CoNIC_unified/best.pth \
  --data-root /data1/llx/CoNICdata --gpu 3 \
  --out-dir ./output/CoNIC_unified/centroid_eval
$PY benchmark_efficiency.py --config ./configs/CoNIC_unified.yml \
  --weight ./output/CoNIC_unified/best.pth --gpu 3 --out ./output/efficiency_apgcc_unified.json