# 基于梯度变化统计的变分优化器实验总结

## 1. 研究目标

本轮实验的目标不是重新设计完整训练系统，而是在现有 `train.py` 的单卡、固定 5 分钟训练预算设定下，验证一个更聚焦的问题：

> 梯度变化统计是否能够作为一种有用的“变分信号”，在不破坏原有优化框架的前提下改善验证 `val_bpb`？

这里的“变分”并不是指严格的变分推断实现，而是把梯度变化幅度视为一种不确定性或局部曲率变化的近似信号，用来调节更新幅度、更新方向或训练调度。

---

## 2. 核心公式与方法设计

### 2.1 主方法：GradVar-AdamW

最初设想是在 AdamW 的二阶统计之外，再显式维护一项梯度变化统计：

```math
d_t = \beta_3 d_{t-1} + (1-\beta_3)(g_t-g_{t-1})^2
```

然后把它作为额外阻尼项加入 AdamW 分母：

```math
\theta_{t+1}
=
(1-\eta\lambda)\theta_t
- \eta \frac{\hat m_t}{\sqrt{\hat v_t + \rho \hat d_t} + \epsilon}
```

其中：

- `m_t` 是一阶动量
- `v_t` 是二阶动量
- `d_t` 是梯度变化的指数滑动平均
- `rho` 控制变化统计对更新的抑制强度

这个设计的出发点是：当相邻步梯度变化剧烈时，说明局部优化地形不稳定，更新应更保守。

### 2.2 变体思路

在主公式无效或开销过大后，又尝试了几类更弱的表达：

1. **AdamW-side damping**
   只在 AdamW 参数组上引入 `d_t`，不动 Muon 主干。

2. **Muon-side scalar damping**
   不保存整块梯度历史，只保存每个矩阵的 RMS 变化统计，用它调节 Muon 更新。

3. **Multiplicative gate**
   不把变化统计放进分母，而是把它变成一个接近 `1` 的门控因子，轻微缩放更新。

4. **Hybrid Muon-AdamW**
   对 `lm_head` 这种大矩阵参数，把 AdamW 方向与 Muon 风格的正交化方向做小比例混合。

5. **Meta-scheduler variants**
   不直接改更新公式，而是让变化统计只去调节：
   - Muon momentum
   - Muon weight decay
   - Muon learning rate

这些变体的共同目标是降低系统开销，并检验“梯度变化作为不确定性信号”本身是否有价值。

---

## 3. 实验设计

### 3.1 固定实验环境

- 数据与评估：由 `prepare.py` 固定，不做修改
- 指标：`val_bpb`，越低越好
- 时间预算：训练阶段固定 300 秒，整体实验超过约 10 分钟视为失败
- 硬件：单张 NVIDIA 5060
- 可改文件：仅 `train.py`

### 3.2 对照方式

实验分成两层：

1. **容量基线**
   先确认当前训练脚本的有效容量区间，避免把“模型太小”误判成“优化器不好”。

2. **优化器实验**
   在当前最好容量基线上，只替换或扩展优化器策略，比较同一时间预算内的 `val_bpb`。

### 3.3 重要基线

| commit | 设置 | val_bpb | 备注 |
|---|---|---:|---|
| `1c27729` | 原始基线 | 1.628552 | 深度过小，模型明显欠容量 |
| `945c757` | `DEPTH=4` 基线 | 1.396539 | 当前最优有效结果 |

`945c757` 说明本项目中最显著的提升首先来自容量修正，而不是优化器微调。

---

## 4. 关键实验结果

### 4.1 与变分优化器直接相关的主要实验

| commit | 方法 | 结果 | 状态 |
|---|---|---:|---|
| `50edbb7` | AdamW 参数组加入梯度变化阻尼 | 1.628637 | discard |
| `5e9992e` | Muon-side scalar gradvar optimizer | 1.748301 | discard |
| `b19f340` | group-level gradvar LR gating | 1.750228 | discard |
| `b16bf22` | 重写后的 gated GradVar-AdamW | 1.759879 | discard |
| `5fcdc09` | selective GradVar-AdamW groups | 1.749533 | discard |
| `1a52dc2` | scalar-only GradVar-AdamW | 1.739078 | discard |
| `646f016` | `lm_head` 上的弱 multiplicative gate | 1.749062 | discard |
| `3ad3d92` | hybrid Muon-AdamW head optimizer | 1.750358 | discard |
| `848a4ab` | variation-controlled Muon momentum | 1.741060 | discard |
| `93f2f68` | variation-controlled Muon weight decay | 1.740790 | discard |
| `ee08d20` | variation-controlled Muon learning rate | 1.738969 | discard |

### 4.2 运行时失败但有信息量的实验

| commit | 方法 | 状态 | 含义 |
|---|---|---|---|
| `fcd3a39` | Muon gradvar damping | crash | 逐元素历史状态导致吞吐崩塌 |
| `0abe953` | Muon RMS-diff gradvar | crash | 吞吐恢复，但总墙钟仍超预算 |
| `1ed83f4` | AdamW gradvar only | crash | 说明即使只作用于 AdamW 支路也可能破坏整体效率 |
| `4e019bb` | late weak hybrid head optimizer | crash | 极弱混合仍难在总运行时约束内稳定完成 |

---

## 5. 实验观察

### 5.1 当前配方下，梯度变化统计不是有用的优化信号

无论采用哪种表达方式，`val_bpb` 都显著差于当前最好基线 `1.396539`。这表明在本任务上：

- 梯度变化率并没有提供有效的“局部不确定性”信号
- 把它转成阻尼、门控、方向混合或元调度，都会伤害训练

### 5.2 负结果不只是系统实现问题

最早的失败确实包含明显的系统代价：

- 保存额外历史状态
- 逐元素引入新统计
- 影响 fused / compiled 路径

但后续更弱、更轻的实验仍然失败：

- 只作用在极小参数组
- 只作用在 `lm_head`
- 只调节 momentum / weight decay / learning rate

这些实验的吞吐基本正常，但结果仍然变差。因此当前结论不能仅归因于“实现太慢”。

### 5.3 Muon 的矩阵思想不能直接迁移

Muon 在当前脚本里有效，说明矩阵参数确实适合结构化更新。但实验显示：

- 直接混合 AdamW 与 Muon 的方向，不会自动变好
- 把 variation statistic 作为 Muon 的外部控制量，也没有带来增益

因此，“参考 Muon 的矩阵优化思路”本身是合理的，但当前这种参考方式并不成立。

---

## 6. 结论

本轮实验给出了一个比较明确的负结果：

> 在本仓库的固定 5 分钟训练预算、当前数据分布、模型规模和优化器配方下，基于梯度变化统计的变分优化器并没有带来正收益。

更具体地说：

1. 把 `(g_t-g_{t-1})^2` 作为额外阻尼项加入 AdamW 分母，效果显著变差。
2. 把这一信号弱化为 gate、方向混合或元调度，结果仍然普遍劣于基线。
3. 参考 Muon 的矩阵优化思想进行混合，也没有改善 `val_bpb`。
4. 当前项目中真正有效的提升首先来自容量修正，而不是这一类优化器设计。

当前最优有效结果仍然是：

- `commit: 945c757`
- setting: `DEPTH=4`
- `val_bpb: 1.396539`

---

## 7. 下一步建议

如果继续推进论文或实验，建议不要再围绕“梯度变化率”这一信号做微调，而应转向更不同的变分表达，例如：

1. **基于梯度能量与参数能量比的先验控制**
2. **对矩阵参数做低秩或谱范数层面的后验近似**
3. **用 KL / 熵风格的简化正则控制 `lm_head` 或 embedding**

也就是说，下一阶段如果还要坚持“变分法”路线，应该换信号，而不是继续修改同一个 `gradient variation` 统计量。
