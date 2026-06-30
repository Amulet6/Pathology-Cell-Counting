# A+D+E on CoNIC — 三版完整结果

Base: APGCC(VGG16-bn+IFI) + **D** DCNv2(2)+Edge(16/0.1) + **E** stain aug, 叠加不同的 **A** 训练配方.
Train: SHHA→CoNIC finetune, 100ep, val selects best. Test: CoNIC test.list, 991 imgs, GT 112545.
CoNIC baseline 参照 (native, 0.5): MAE ~12 (11.99–12.93), F1@12 ~0.793; D Edge-only 11.01; E stain+domainTh 11.48.

## 三版主对比 (CoNIC test, 诚实协议)

| 版本 | A 训练配方 | val→test 全局 MAE | A 密度自适应 MAE | E 域阈值 MAE | F1@12 | 0.5处偏差 |
|---|---|---:|---:|---:|---:|---:|
| ① EOS0.10 | EOS↓↓+focalγ2+τ0.10 | 21.21 (thr0.55) | 9.65 (Δ=0 无增益) | 16.68 | 0.777 | 过计 +36% |
| ② EOS0.25 | EOS↓+focalγ2+τ0.10 | 15.19 (thr0.50) | 15.19 (Δ=0 无增益) | 15.07 | 0.783 | 欠计 −11% |
| **③ D+E clean** | **关闭(EOS0.5/focal0/τ0.05)** | **11.87 (thr0.45)** | **10.37 (Δ=+1.0 有增益!)** | 11.73 | **0.7966** | 欠计 −7.7% |
| baseline 参照 | — | ~12 | — | — | ~0.793 | 平衡 |

注: A 密度自适应 = density_threshold.py 2-fold CV 诚实报告 (GLOBAL-CV→DENSITY-CV).
①②的 9.65/15.19 是 test-CV 单阈值; ③ DENSITY-CV 10.37 真正用上了密度分档.

## ③ D+E clean 细节 (主结果)

- 训练: 5h08m, best=ep40 (val MAE 3.452, 三版最低).
- fixed 0.5: MAE 12.25, F1@12 0.7966 (P0.830/R0.766), −7.7%.
- 全局阈值: test@0.40=11.56, val-best(0.45)→test=11.87, F1@12 ~0.794–0.797.
- **A 密度自适应: GLOBAL-CV 11.37 → DENSITY-CV 10.37 (Δ=+1.0), per-bin thr=0.5/0.3/0.1 (密度越高阈值越低,
  符合机理!), oracle 5.63.**
- E 域阈值: MAE 11.73, F1@12 0.7962 (阈值 consep0.35/crag0.45/dpath0.35/glas0.55/pannuke0.4).

## 核心结论 (后天汇报口径)

1. **主结果 = D+E clean + A 密度自适应阈值: MAE 10.37, F1@12 0.797**, 同时**优于 baseline (MAE~12, F1 0.793)**:
   计数误差降 ~13%, 定位 F1 略升. 干净的正向结果.
2. **D+E 的结构改进有效**: DCNv2+Edge+stain 让 F1@12 (0.794–0.797) 达到/超过 baseline (0.793),
   计数追平 baseline; 这是结构层面的真增益.
3. **A 的密度自适应阈值在干净基座上 work**: DENSITY-CV 比单一全局阈值降 1.0 MAE (11.37→10.37),
   per-bin 阈值 0.5/0.3/0.1 完全符合"高密度区降阈值"机理. 对比①②(被激进配方破坏时 Δ=0), 说明
   **A 的后处理需要一个未被激进训练配方扭曲的置信度分布才能发挥**.
4. **A 的激进训练配方(EOS↓+focal)在 native 好基座上有害**: ①过计(+36%)②欠计(−11%), 诚实 MAE 都不如 baseline,
   且破坏了密度自适应的前提. 这与 Direction-A 自己结论一致 ("native 上配方≈中性甚至有害, 方向与 unified 相反").

## 一句话
A+D+E 在 CoNIC 上真正 net-positive 的形态是 **D+E 结构改进 + A 密度自适应后处理**(MAE 10.37 / F1 0.797, 超 baseline),
而**不是** A 的激进训练配方(EOS↓+focal, 在好基座上有害). A 的贡献体现在后处理校准, 不在训练配方.
