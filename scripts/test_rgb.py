import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from env.Breakout import BreakoutEnv

if __name__ == "__main__":
    Env = BreakoutEnv(render=True)
    obs, info = Env.env.reset()
    end = False
    while not end:
        action = Env.env.action_space.sample()
        obs, reward, terminated, truncated, info = Env.step(action)
        # for key, value in info.items():
        #     print(f"{key}: {value} (类型: {type(value)})")

        if reward > 0:
            print(f"获得奖励: {reward}")
        
        if terminated or truncated:
            print("游戏结束")
            end = True
            obs, info = Env.reset()

    Env.close()