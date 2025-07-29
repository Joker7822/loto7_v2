import gymnasium as gym
import numpy as np

class LotoEnv(gym.Env):
    def __init__(self, historical_numbers):
        super().__init__()
        self.historical_numbers = historical_numbers
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(37,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(37,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.state = np.zeros(37, dtype=np.float32)
        return self.state, {}

    def step(self, action):
        # action: [0.1, 0.8, ..., 0.05] → 上位7個を選ぶ
        top_indices = np.argsort(action)[-7:]
        selected_numbers = set(top_indices + 1)

        # 簡単な報酬: 最近の本数字と何個一致したか
        recent = self.historical_numbers[-7:] if len(self.historical_numbers) >= 7 else self.historical_numbers
        reward = sum(n in selected_numbers for n in recent)

        done = True
        info = {"selected": selected_numbers}
        return self.state, reward, done, False, info
