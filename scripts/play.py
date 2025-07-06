import torch
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from env.Breakout import BreakoutEnv
from network.DQN import DQN, FrameStack

if __name__ == "__main__":
    # 修正模型路径
    model_path = "model/dqn_breakout_episode_2000.pth"

    # 环境设置 - 使用和test_rgb.py相同的方式
    Env = BreakoutEnv(render=True,frameskip = 4)  # 保持和test一样的设置
    num_actions = Env.env.action_space.n
    
    # 动作名称映射（Breakout的动作空间）
    action_names = {
        0: "NOOP",      # 无操作
        1: "FIRE",      # 发射球
        2: "RIGHT",     # 向右移动
        3: "LEFT"       # 向左移动
    }

    # 初始化帧堆叠
    frame_stack = FrameStack(num_frames=4)

    # 初始化网络
    input_shape = (4, 84, 84)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    policy_net = DQN(input_shape, num_actions).to(device)

    # 加载权重
    try:
        checkpoint = torch.load(model_path, map_location=device)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        policy_net.eval()
        print("模型加载成功")
        print(f"模型探索率: {checkpoint.get('exploration_rate', 'N/A')}")
    except FileNotFoundError:
        print(f"模型文件未找到: {model_path}")
        exit()
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()

    # 开始游玩
    obs, info = Env.reset()  # 使用和test相同的方式
    state = frame_stack.reset(obs)
    total_reward = 0
    step = 0

    print("开始AI游戏...")
    print(f"动作空间大小: {num_actions}")
    print("=" * 50)

    # 生命值和强制发球相关变量
    prev_lives = info.get('lives', 5)  # 初始生命值
    life_lost_recently = False  # 最近是否失去生命

    while True:
        # 转为张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
        
        # 输出AI的动作选择和Q值信息
        action_name = action_names.get(action, f"未知动作{action}")
        max_q_value = q_values.max().item()

        # 如果刚失去生命，强制发球
        if life_lost_recently:
            action = 1  # 强制FIRE
            life_lost_recently = False
        
        
        # 每10步输出一次详细信息
        if step % 10 == 0:
            print(f"步骤 {step:4d}: 动作 {action} ({action_name:5s}) | Q值: {max_q_value:.3f} | 总奖励: {total_reward}")
            # 输出所有动作的Q值
            q_values_str = " | ".join([f"{action_names.get(i, f'A{i}')}: {q_values[0][i].item():.3f}" 
                                     for i in range(num_actions)])
            print(f"         所有Q值: {q_values_str}")
        
        # 执行动作 - 使用和test相同的方式
        next_obs, reward, terminated, truncated, info = Env.step(action)
        next_state = frame_stack.step(next_obs)
        
        current_lives = info.get('lives', 5)
        if current_lives < prev_lives:
            life_lost_recently = True
            prev_lives = current_lives

        total_reward += reward
        step += 1

        state = next_state

        if reward > 0:
            print(f"*** 步骤 {step}: 获得奖励 {reward!r}! 动作: {action} ({action_name}) ***")
        
        # 当AI选择了新的动作时也输出
        if step > 1:  # 避免第一步的比较
            prev_action = getattr(Env, '_last_action', None)
            if prev_action != action:
                print(f"    动作变化: {action_names.get(prev_action, 'N/A')} -> {action_name}")
        
        # 记录当前动作
        Env._last_action = action
        
        if terminated or truncated:
            print("=" * 50)
            print(f"游戏结束!")
            print(f"总奖励: {total_reward}")
            print(f"总步数: {step}")
            print(f"平均奖励: {total_reward/step:.4f}")
            break

    Env.close()