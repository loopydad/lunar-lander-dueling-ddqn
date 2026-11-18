# LunarLander · Dueling Double DQN

用 PyTorch 实现 **Dueling Double DQN**，在 Gymnasium 的 `LunarLander-v3` 环境中学习登月舱控制。项目涵盖经验回放、ε-greedy 探索、目标网络、TD 学习及独立评估，保留单文件实现，便于对照算法阅读。

原始算法代码主要由 **Gary / [loopydad](https://github.com/loopydad)** 独立编写。Codex 协助完成此次工程整理、运行验证和文档；完整来源与改动见 [SOURCE.md](SOURCE.md)。

## 方法与数据流

每个状态包含 8 个观测：位置、速度、角度、角速度以及两个腿部接触标志。动作有 4 个：不点火、左侧引擎、主引擎、右侧引擎。网络输出动作价值，不是动作概率。

```text
8维观测 → 共享特征层(128 → 128) → 基线分支 V: 1维
                                 → 优势分支 A: 4维
                     ↓
         Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
```

Dueling 将共同状态水平和动作差异分开表示；这里 V 是输出 Q 的动作均值，不是最优状态价值 max Q。Double DQN 在下一状态由在线网选择动作、目标网估值，以缓解 max 选择偏差：

$$
a^*=\arg\max_a Q_\theta(s',a),\qquad
 y=r+\gamma(1-\text{terminated})Q_{\theta^-}(s',a^*).
$$

用在线网预测实际执行动作的 Q，通过 MSE 拟合上述停止梯度的目标。外部时间截断仍保留自举，真正终止才清零未来项。

```text
ε-greedy选动作 → 环境走一步 → 保存转移 → 随机抽64条 → 更新一次在线网
                    ↑                                   ↓
                    └──────── 在当前状态继续交互 ──────────┘
                 每1000个环境步复制在线参数到目标网
```

经验池不足 64 条时跳过训练，累计第 64 条保存后首次更新。经验池跨回合保留；训练目标中的下一动作仅用于估值，不会在环境中执行。

## 安装

建议 Python 3.11，在独立虚拟环境运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install swig
python -m pip install -r requirements.txt
```

Gymnasium 的 Box2D 依赖在部分平台可能需要本地构建工具；详见 [LunarLander 官方文档](https://gymnasium.farama.org/environments/box2d/lunar_lander/)。运行版本见 [验证记录](docs/validation.md)。

## 运行

### 先验证完整流程

```bash
python -m unittest -v
python lunar_lander.py --episodes 12 --max-steps 200 --eval-episodes 3 \
  --seed 42 --output-dir runs/smoke
```

这是缩短回合上限的 smoke run，用于检查交互、反向传播、保存和评估，不是学习效果实验。实际记录见 [docs/validation.md](docs/validation.md)。

### 正式训练

```bash
python lunar_lander.py --episodes 5000 --seed 42 --output-dir runs/seed42
```

默认 CPU，无需图形界面；可通过 `--device cuda` 或 `--device mps` 指定设备。小型网络默认单线程。最近100个训练回合平均奖励达到200时提前停止，再使用新环境、不同种子和关闭探索的策略独立评估10局。该训练停止条件不自动等于稳定解决任务。

每次实验使用不同的 `--output-dir`，同名目录会覆盖输出：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 运行参数、关键超参数与主要依赖版本 |
| `training.csv` | 每回合回报、步数、探索率与平均训练损失 |
| `learning_curve.png` | 原始回报和20回合移动平均 |
| `model.pt` | 在线网络最终权重，仅用于推理 |
| `metrics.json` | 交互步、梯度更新数、目标同步数及独立评估回报 |

### 加载权重评估或演示

```bash
python lunar_lander.py --eval-only --checkpoint runs/seed42/model.pt \
  --eval-episodes 20 --seed 100 --output-dir runs/eval42

# 在有桌面窗口的电脑上演示3局
python lunar_lander.py --eval-only --checkpoint runs/seed42/model.pt \
  --eval-episodes 3 --render --output-dir runs/demo42
```

评估不更新参数，结果保存为 `evaluation.json`。权重不含优化器和经验池，不支持断点续训。评估默认使用 `seed + 100000` 起的环境种子；正式报告应采用多个训练种子和更多独立评估回合。固定种子帮助复查，不保证跨硬件、版本的逐位一致。

## 默认超参数

| 参数 | 数值 |
| --- | ---: |
| 学习率 / 优化器 | 1e-4 / Adam |
| 折扣因子 | 0.99 |
| Batch size / 回放容量 | 64 / 50000 |
| 探索率 | 1.0开始，每回合×0.995，最低0.01 |
| 目标网硬同步 | 每1000个环境步 |
| 梯度范数上限 | 1.0 |
| 最多训练回合 / 每回合步数 | 5000 / 1000 |

## 项目结构

```text
lunar_lander.py           可配置的训练、保存与评估入口
original/lunar_lander.py  原始实现快照
requirements.txt         运行依赖
test_lunar_lander.py     核心算法语义测试
SOURCE.md                作者与整理改动
docs/validation.md      实际运行证据
```

## 项目经历表述

**基于 Dueling Double DQN 的登月舱控制**：基于 PyTorch 编写深度强化学习控制代码，结合 Dueling 价值分解、Double DQN 目标、经验回放和 ε-greedy 探索，完成 LunarLander 环境的交互与策略学习流程；在 AI 辅助下补充参数配置、日志、模型保存及独立评估。

完整训练性能以实验记录为准。目前公开验证为短程流程运行，未声称达到200分或稳定着陆。

## 参考

- [Double DQN](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures](https://arxiv.org/abs/1511.06581)
- [Gymnasium：终止、截断与自举](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
