import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from collections import deque

# ==========================================
# 1. 超参数设置 
# ==========================================
LR = 1e-4  # 学习率
GAMMA = 0.99  # 折扣因子 (越接近1越看重长远收益)
BATCH_SIZE = 64  # 每次从经验回放中采样的样本数
MEMORY_CAPACITY = 50000  # 经验回放池的容量
MIN_EPSILON = 0.01  # 最小探索率
EPSILON_DECAY = 0.995  # 探索率衰减系数
TARGET_UPDATE_STEPS = 1000  # 每隔多少个 step 更新一次目标网络
MAX_EPISODES = 5000  # 最大训练回合数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)
# ==========================================
# 2. 经验回放池 (Experience Replay)
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        # 使用 deque 实现固定容量的队列，满了会自动覆盖旧数据
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 随机抽取一个批次的经验，打破数据间的相关性
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


# ==========================================
# 3. Dueling DQN 神经网络模型
# ==========================================
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        # 提取公共特征的网络层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # 【Dueling 特色 1】Value Stream: 评估当前状态有多好 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 【Dueling 特色 2】Advantage Stream: 评估在这个状态下，每个动作比平均水平好多少 A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        features = self.feature_layer(state)

        V = self.value_stream(features)
        A = self.advantage_stream(features)

        # 【Dueling 聚合层】 Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # 减去均值是为了保证网络的可识别性 (Identifiability)，使 V(s) 的意义更纯粹
        Q = V + A - A.mean(dim=1, keepdim=True)
        return Q


# ==========================================
# 4. 智能体 (Agent) 定义
# ==========================================
class Agent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.epsilon = 1.0  # 初始 100% 随机探索

        # 实例化在线网络 (Online Network) 和目标网络 (Target Network)
        self.online_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        # 初始化时，两个网络权重完全一致
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # 目标网络只评估，不进行梯度更新

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)

    def choose_action(self, state, evaluate=False):
        # 评估模式下不探索，只利用
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.online_net(state_tensor)
            return q_values.argmax().item()

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return  # 经验池里的数据还不够一个 Batch 时，先不训练

        # 1. 从回放池采样
        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)

        state = torch.FloatTensor(state).to(device)
        action = torch.LongTensor(action).unsqueeze(1).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        # 2. 计算当前的 Q 值
        # gather 的作用是提取我们实际执行过的那个动作对应的 Q 值
        q_values = self.online_net(state).gather(1, action)

        # 3. 计算目标 Q 值 (Target Q) - 【此处是 Double DQN 的核心!】
        with torch.no_grad():
            # [DDQN 步骤 A]: 使用 在线网络 选出下一个状态中 Q 值最大的 动作
            next_actions = self.online_net(next_state).argmax(dim=1, keepdim=True)
            # [DDQN 步骤 B]: 使用 目标网络 评估这个动作的 Q 值
            # 这有效避免了标准 DQN 自己选自己评导致的“盲目乐观 (过高估计)”
            next_q_values = self.target_net(next_state).gather(1, next_actions)

            # 如果游戏结束，未来收益为0；否则为 reward + 折扣后的未来收益
            target_q_values = reward + (1 - done) * GAMMA * next_q_values

        # 4. 计算损失并更新在线网络
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸，让训练更稳定
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self):
        # 硬更新：直接把在线网络的参数复制给目标网络
        self.target_net.load_state_dict(self.online_net.state_dict())


# ==========================================
# 5. 辅助函数：绘制平滑曲线
# ==========================================
def plot_learning_curve(rewards):
    # 使用滑动平均来平滑曲线，看起来更直观
    window_size = 20
    smoothed_rewards = [np.mean(rewards[max(0, i - window_size):i + 1]) for i in range(len(rewards))]
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label='Raw Reward', color='gray')
    plt.plot(smoothed_rewards, label=f'Smoothed Reward (window={window_size})', color='blue')
    plt.title('Dueling DDQN - LunarLander-v3 Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.show()


# ==========================================
# 6. 主程序：训练与可视化展示
# ==========================================
if __name__ == "__main__":
    env = gym.make("LunarLander-v3")

    state_dim = env.observation_space.shape[0]  # 状态维度: 8
    action_dim = env.action_space.n  # 动作维度: 4 (无动作, 左引擎, 主引擎, 右引擎)

    agent = Agent(state_dim, action_dim)
    rewards_history = []
    global_step = 0

    print("开始训练，请耐心等待 (约需几分钟)...")
    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # 1. 选动作
            action = agent.choose_action(state)

            # 2. 与环境交互
            next_state, reward, done, truncated, _ = env.step(action)

            # 3. 存入经验池
            agent.memory.push(state, action, reward, next_state, done or truncated)

            # 4. 学习更新
            agent.update()

            # 5. 统计全局 step，并按 step 硬更新目标网络
            global_step += 1
            if global_step % TARGET_UPDATE_STEPS == 0:
                agent.update_target_network()

            state = next_state
            episode_reward += reward

        # 探索率衰减
        agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)
        rewards_history.append(episode_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1:3d} | Reward: {np.mean(rewards_history[-100:]):6.2f} | Epsilon: {agent.epsilon:.3f}")

        # 如果连续几局平均得分超过 200 (算作通关)，可以提前结束训练
        if len(rewards_history) >= 100 and np.mean(rewards_history[-100:]) >= 200:
            print(f"\n太棒了！在第 {episode + 1} 回合达到目标，提前结束训练！")
            break

    env.close()

    # --- 展示环节 ---
    print("\n训练完成，正在绘制学习曲线...")
    plot_learning_curve(rewards_history)

    print("\n准备就绪，即将展示智能体控制登月舱着陆！")
    # 使用 render_mode="human" 重新开启环境进行肉眼观察
    test_env = gym.make("LunarLander-v3", render_mode="human")

    for test_ep in range(3):  # 演示 3 次
        state, _ = test_env.reset()
        test_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            # 评估模式，每次选择最优动作
            action = agent.choose_action(state, evaluate=True)
            state, reward, done, truncated, _ = test_env.step(action)
            test_reward += reward
        print(f"演示回合 {test_ep + 1} 总得分: {test_reward:.2f}")

    test_env.close()
    print("演示结束，感谢观看！")