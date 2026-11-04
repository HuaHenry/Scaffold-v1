# 04 OOD 泛化分析

优先级 P0 ｜ 对应假设：H3 ｜ 依赖：实验 03 的 12 个 run ｜ 公共约定见 `00`

## 1. 目标

量化各条件从分布内（`id`）到分布外（三个 OOD 划分）的性能下降，并检验：process 对 OOD 的增益是否**大于**对分布内的增益（H3）。

## 2. 意义

"只见成品的模型泛化更差"才是形成缺口的实质性主张；分布内的小差异不足以支撑宏大叙事。三种 OOD 检验不同能力：

| 划分 | 检验的能力 | 构造 |
| --- | --- | --- |
| `combo_ood` | 新组合 | cottage+tree、hut+tower、bridge+dune、watchtower+pond、arch+garden、cottage+watchtower |
| `scale_ood` | 新尺度（更长的悬挑与承重链） | tower / bridge / arch / cottage / watchtower 的 big 版本 |
| `multi_ood` | 多物体（干预只影响其中一个） | tower+bridge+tree 等四组 |

## 3. 实验设计

无需新训练，使用 `runs/main/*_s{0,1,2}/eval_samples.npz`，另加一个新评测集（尺度外推）。

1. **OOD 差距**：每个条件的 `gap = metric(id) - metric(OOD)`，指标取连锁 F1、`cell_acc_mask`、排序准确率。
2. **H3 检验**：对每个 OOD 划分，`Δ_OOD = metric(process) - metric(state)` 与 `Δ_ID`（在 id 划分），检验 `Δ_OOD - Δ_ID`。因来自不同样本集，用分层 bootstrap（在 OOD 与 id 内分别重采样），取差的差的 95% CI；(process,partial)、(process,shuffle) 各做一次。
3. **按难度分段**：用真值连锁大小（`n_changed` 分箱 0–1、2–4、5–9、>=10）画连锁 F1 随难度下降的曲线。
4. **尺度外推曲线**：用 `datasets.sample_scene` 以固定种子生成 tower 高度 4…16、bridge 跨度 4…14 的单结构评测集（每个设定 32 个样本，`candidate_interventions` 取承重块，真值生成与验证逻辑复用 `sample_bench_item`，排除与训练成品重复的结构），画"高度 / 跨度 -> 连锁 F1"，并标出训练分布边界（tower 高 4–7、bridge 跨 4–8）。
5. **组件对照**：`combo_ood` 每个样本的两个组成场景，比较"组合后的性能"与"两个单独场景在 `id` 上性能的均值"，量化组合带来的额外下降。

## 4. 实现（本实验独有部分）

```bash
python -m exp.analysis.ood_gap     --exp main --out results/ood_gap.json --table results/tables/ood_gap.md
python -m exp.analysis.h3_test     --exp main --out results/h3.json
python -m exp.analysis.scale_sweep --runs runs/main --out results/scale_sweep.json
```

`h3_test.py` 输出每个（比较对, OOD 划分, 指标）的 `delta_id, delta_ood, diff, ci_low, ci_high`。产出图表：Fig.5、Table 3 的 OOD 部分。

## 5. 验收标准

1. 各条件 `id` 的连锁 F1 >= OOD 划分（不成立需解释，例如 multi_ood 因干预被稀释而分数偏高）。
2. 尺度外推评测集中，训练分布内（tower 高 4–7）的 oracle_reseed 精确匹配 >= 0.95，保证评测本身有效。
3. 所有 CI 由固定种子 bootstrap 产生，可复现。

## 6. 预期结果（先验判断）

- 所有条件在 `scale_ood` 下降最明显（平均连锁 8.6 格，对比 `id` 的 2.7 格）；`multi_ood` 的连锁 F1 可能不低，但 `cell_acc_mask` 被静止结构抬高。
- 连锁 F1 随结构高度 / 跨度增长而下降，训练分布边界之外下降更快；若 process 在边界外下降更慢，就是 H3 最直接的证据。
- H3 整体：先验约 30% 成立（process 的 OOD 增益 > id 增益且 CI 不含 0），多数情况差异不显著；不成立则如实报告，并把本实验改写为"OOD 下降的刻画"。

## 7. 风险与备选

- `multi_ood` 的 `rank_valid` 仅约 65%，排序检验以 `id` / `combo_ood` / `scale_ood` 为主，表中标样本数。
- OOD 每划分只有 128 个样本，CI 过宽时用新种子再生成一份 bench 合并。
