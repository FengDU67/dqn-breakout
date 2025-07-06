import gymnasium
import ale_py
import matplotlib.pyplot as plt
import numpy as np
import random

gymnasium.register_envs(ale_py)

class BreakoutEnv:
    def __init__(self, render=True, mode=0, difficulty=0, frameskip=1):
        # self.mode = random.choice([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44])
        # self.difficulty = random.choice([0, 1])
        self.mode = mode
        self.difficulty = difficulty
        self.frameskip = frameskip
        self.env = gymnasium.make(
            "ALE/Breakout-v5", 
            render_mode="rgb_array",
            mode=self.mode,
            difficulty=self.difficulty,
            frameskip=self.frameskip,  # 每4帧采样一次
        )
        
        print(f"环境模式: {self.mode}, 难度: {self.difficulty}")
        self.render_enabled = render
        if render:
            plt.ion()
            self.fig, self.ax = plt.subplots()
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # obs和frame完全相同
        
        if self.render_enabled:
            frame = self.env.render()
            if frame is not None:
                # # 打印frame的详细信息
                # print(f"Frame类型: {type(frame)}")
                # print(f"Frame形状: {frame.shape}")
                # print(f"Frame数据类型: {frame.dtype}")
                # print(f"Frame数值范围: {frame.min()} - {frame.max()}")
                # print(f"Frame前几个像素: {frame[0, 0]}")  # 第一个像素的RGB值
                # Frame类型: <class 'numpy.ndarray'>
                # Frame形状: (210, 160, 3)
                # Frame数据类型: uint8
                # Frame数值范围: 0 - 200
                # Frame前几个像素: [0 0 0]

     
                self.ax.clear()
                self.ax.imshow(frame)
                self.ax.set_title("Breakout Game")
                self.ax.axis('off')
                plt.pause(0.01)
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        if self.render_enabled:
            plt.ioff()
            plt.close()
        self.env.close()

