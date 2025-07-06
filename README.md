# DQN for Atari Breakout

基于深度Q网络(DQN)的Atari Breakout游戏强化学习项目。

## 项目结构

```
rl-ale-breakout/
├── env/
│   └── Breakout.py          # Breakout环境封装
├── network/
│   └── DQN.py              # DQN网络和训练逻辑
├── scripts/
│   ├── train.py            # 训练脚本
│   ├── play.py             # 模型测试脚本
│   └── test_rgb.py         # 环境测试脚本
└── README.md
```

## 环境要求

- Python 3.8+
- PyTorch
- Gymnasium[atari]
- ALE-py
- OpenCV
- Matplotlib

## 安装依赖

```bash
pip install torch torchvision torchaudio
pip install gymnasium[atari]
pip install ale-py
pip install opencv-python
pip install matplotlib
pip install autorom
AutoROM --accept-license
```

## 使用方法

### 测试环境

```bash
python scripts/test_rgb.py
```

### 训练模型

```bash
python scripts/train.py
```


### 使用训练好的模型

```bash
python scripts/play.py
```

## 特性

- 基于DQN的强化学习算法
- 帧堆叠(Frame Stacking)技术
- 经验回放(Experience Replay)
- 目标网络(Target Network)
- 强制发球机制，避免AI不行动
- 生命值惩罚和存活时长奖励

## 训练策略

- 失去生命惩罚: -1.0
- 存活时长奖励: 每100步 +0.1
- 强制发球: 防止AI选择不动策略
- 探索率衰减: 线性衰减保持探索
