import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import cv2

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from env.Breakout import BreakoutEnv

# 设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 预处理图像
def preprocess(frame):
    # 转灰度图
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 调整大小
    frame = cv2.resize(frame, (84, 84))
    # 归一化
    frame = frame.astype(np.float32) / 255.0
    return frame

# 帧堆叠类
class FrameStack:
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, frame):
        # 重置时用相同帧填充
        processed_frame = preprocess(frame)
        for _ in range(self.num_frames):
            self.frames.append(processed_frame)
        return np.array(self.frames)
    
    def step(self, frame):
        # 添加新帧
        processed_frame = preprocess(frame)
        self.frames.append(processed_frame)
        return np.array(self.frames)

# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.feature_size(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def feature_size(self, input_shape):
        return self._forward_conv(torch.zeros(1, *input_shape)).view(1, -1).size(1)

    def _forward_conv(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # 输出Q值

# 经验回放
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class GPUReplayMemory:
    def __init__(self, capacity, state_shape, device, dtype=torch.float32):
        """
        优化的GPU经验回放缓冲池
        capacity: 缓冲池容量
        state_shape: 状态形状 (4, 84, 84)
        device: GPU设备
        dtype: 数据类型，可以用float16节省显存
        """
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        print(f"初始化GPU缓冲池，容量: {capacity}")
        
        # 使用float16可以节省一半显存
        if dtype == torch.float16:
            print("使用半精度浮点数 (float16) 节省显存")
        
        try:
            # 预分配GPU张量
            self.states = torch.zeros((capacity, *state_shape), dtype=dtype, device=device)
            self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
            self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
            self.next_states = torch.zeros((capacity, *state_shape), dtype=dtype, device=device)
            self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
            
            # 检查显存使用
            memory_used = torch.cuda.memory_allocated(device) / (1024**3)
            print(f"GPU缓冲池创建成功，使用显存: {memory_used:.2f} GB")
            
        except RuntimeError as e:
            print(f"GPU缓冲池创建失败: {e}")
            print("建议减小缓冲池大小或使用混合存储方案")
            raise
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        # 转换并存储
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state.copy())
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state.copy())
            
        self.states[self.position] = state.to(self.device, dtype=self.states.dtype)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state.to(self.device, dtype=self.next_states.dtype)
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """采样batch"""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        # 返回时转换回float32用于计算
        batch_states = self.states[indices].float()
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices].float()
        batch_dones = self.dones[indices]
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
    
    def __len__(self):
        return self.size


# DQN Agent
class DQNAgent:
    def __init__(self, input_shape, num_actions, buffer_size=15000, use_gpu_buffer=True):
        self.input_shape = input_shape
        self.num_actions = num_actions

        # 根据显存情况选择缓冲池
        if use_gpu_buffer and torch.cuda.is_available():
            try:
                # 尝试创建大容量GPU缓冲池
                self.memory = GPUReplayMemory(
                    capacity=buffer_size, 
                    state_shape=input_shape, 
                    device=device,
                    dtype=torch.float16  # 使用半精度节省显存
                )
                print(f"使用GPU缓冲池，容量: {buffer_size}")
            except RuntimeError:
                # 如果显存不足，降级到CPU
                print("GPU显存不足，使用CPU缓冲池")
                self.memory = ReplayMemory(3500)
        else:
            self.memory = ReplayMemory(3500)

        self.policy_net = DQN(input_shape, num_actions).to(device)
        self.target_net = DQN(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)
        # 基于episode的线性衰减
        self.exploration_start = 1.0
        self.exploration_rate = self.exploration_start
        self.exploration_end = 0.1
        self.exploration_episodes = 1000  # 1000个episode内从1.0衰减到0.05
        self.episode_count = 0
        self.steps_done = 0
    
    def update_exploration_rate(self):
        """基于episode数更新探索率"""
        if self.episode_count < self.exploration_episodes:
            # 线性衰减
            progress = self.episode_count / self.exploration_episodes
            self.exploration_rate = self.exploration_start - progress * (self.exploration_start - self.exploration_end)
        else:
            self.exploration_rate = self.exploration_end
        
        self.episode_count += 1

    def select_action(self, state):
        # ϵ-贪心策略
        if random.random() < self.exploration_rate:
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def train(self, batch_size=32):
        # 样本不足时直接返回
        if len(self.memory) < batch_size:
            return None

        # 检查缓冲池类型并相应处理
        if isinstance(self.memory, GPUReplayMemory):
            # GPU缓冲池返回的已经是GPU张量
            states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
            # actions需要增加维度用于gather操作
            actions = actions.unsqueeze(1)
        else:
            # CPU缓冲池需要转换
            batch = self.memory.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # 转为张量，并去除多余维度
            states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            states = states.squeeze(1) if states.dim() == 5 else states
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
            next_states = next_states.squeeze(1) if next_states.dim() == 5 else next_states
            dones = torch.tensor(dones, dtype=torch.bool).to(device)

        # 计算当前 Q 值
        q_values = self.policy_net(states).gather(1, actions).squeeze()

        # 计算下一时刻最大 Q 值
        next_q_values = self.target_net(next_states).max(1)[0].detach()

        # 计算目标 Q 值
        target_q_values = rewards + (0.99 * next_q_values * ~dones)

        # 优化
        # loss = F.mse_loss(q_values, target_q_values)
        # 使用Smooth L1 Loss
        loss = F.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()


        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filepath):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
        }, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.exploration_rate = checkpoint['exploration_rate']

# 训练函数
def train_dqn():
    # 环境设置
    env = BreakoutEnv(render=False, frameskip=1)  # 训练时关闭渲染提高速度

    # 获取动作空间大小
    num_actions = env.env.action_space.n
    print(f"动作空间大小: {num_actions}")
    
    # 初始化帧堆叠和智能体
    frame_stack = FrameStack(num_frames=4)
    agent = DQNAgent(input_shape=(4, 84, 84), num_actions=num_actions, buffer_size=20000, use_gpu_buffer=True)
    action_repeat = 4  # 每个动作重复4次
    
    # 训练参数
    num_episodes = 2001
    target_update_freq = 2000
    save_freq = 500
    
    episode_rewards = []
    losses = []
    
    for episode in range(num_episodes):
        # 重置环境
        obs, info = env.reset()
        state = frame_stack.reset(obs)
        total_reward = 0
        steps = 0
        
        # 生命值和强制发球相关变量
        prev_lives = info.get('lives', 5)  # 初始生命值
        consecutive_noop = 0  # 连续NOOP计数
        survival_time = 0  # 存活时间
        life_lost_recently = False  # 最近是否失去生命
        
        while True:
            # 选择动作
            action = agent.select_action(state)
            
            # 强制发球逻辑
            original_action = action
            
            # 1. 如果刚失去生命，强制发球
            if life_lost_recently:
                action = 1  # 强制FIRE
                life_lost_recently = False
            
            # 2. 检测连续NOOP并强制发球
            if action == 0:
                consecutive_noop += 1
                if consecutive_noop > 30:  # 连续30步NOOP就强制发球
                    action = 1  # 强制FIRE
                    consecutive_noop = 0
            else:
                consecutive_noop = 0

            # 手动重复动作action_repeat次
            episode_reward = 0
            current_lives = prev_lives
            
            for repeat_step in range(action_repeat):
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                survival_time += 1  # 增加存活时间
                
                # 检查生命值变化
                current_lives = info.get('lives', current_lives)
                if current_lives < prev_lives:
                    # 失去生命的惩罚
                    life_penalty = -1.0  # 失去一条命惩罚1分
                    episode_reward += life_penalty
                    life_lost_recently = True
                    # print(f"Episode {episode}, Step {steps}: 失去生命! ({prev_lives} -> {current_lives}), 惩罚: {life_penalty}")
                    prev_lives = current_lives
                
                if terminated or truncated:
                    break
             
            done = terminated or truncated
            next_state = frame_stack.step(next_obs)

            # 计算修改后的奖励
            modified_reward = episode_reward
            
            # 1. 存活时长奖励 (每存活100步给小奖励)
            if survival_time % 100 == 0 and survival_time > 0:
                survival_bonus = 0.1
                modified_reward += survival_bonus
                # print(f"Episode {episode}, Step {steps}: 存活奖励 +{survival_bonus}")
            
            # 2. 惩罚长时间不动 (原始动作是NOOP且连续多次)
            if original_action == 0 and consecutive_noop > 15:
                noop_penalty = -0.01
                modified_reward += noop_penalty
                # print(f"Episode {episode}, Step {steps}: NOOP惩罚 {noop_penalty}")
            
            # 3. 鼓励积极行动 (非NOOP动作给小奖励)
            if original_action != 0:
                action_bonus = 0.001
                modified_reward += action_bonus
            
            # 存储经验 (使用修改后的奖励)
            agent.memory.push(state, original_action, modified_reward, next_state, done)

            # 训练
            loss = agent.train()
            if loss is not None:
                losses.append(loss)
            
            # 更新目标网络
            if agent.steps_done % target_update_freq == 0:
                agent.update_target_net()
                print(f"目标网络已更新 (步骤: {agent.steps_done})")
            
            state = next_state
            total_reward += modified_reward  # 使用修改后的奖励
            steps += 1
            agent.steps_done += 1
            
            if done:
                # 游戏结束时的额外信息
                final_lives = info.get('lives', 0)
                print(f"Episode {episode} 结束: 最终生命值: {final_lives}, 存活时间: {survival_time}, 总奖励: {total_reward:.2f}")
                break
        
        episode_rewards.append(total_reward)
        agent.update_exploration_rate()
        
        # 打印进度
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episode {episode}, 平均奖励: {avg_reward:.2f}, "
                  f"探索率: {agent.exploration_rate:.3f}, 平均损失: {avg_loss:.4f}")
        
        # 保存模型
        if episode % save_freq == 0 and episode > 0:
            agent.save_model(f"dqn_breakout_episode_{episode}.pth")
            print(f"模型已保存: episode {episode}")
    
    env.close()
    return agent, episode_rewards, losses

