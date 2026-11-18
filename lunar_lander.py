import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import csv
import json
from pathlib import Path
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

device = torch.device("cpu")
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

        # 状态共享基线：等于最终各动作Q值的算术均值，不是V*。
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 原始动作优势输出；减均值后才具有相对平均水平的含义。
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
        # 减均值固定基线与中心化优势的分工；原始A整体平移仍不影响Q。
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
        self.target_net.eval()  # eval不冻结梯度；下面显式冻结，目标也在no_grad内计算。
        self.target_net.requires_grad_(False)

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
            # 分开选择与估值以缓解max高估；两个网络仍相关，不保证完全消除。
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
        return loss.item()

    def update_target_network(self):
        # 硬更新：直接把在线网络的参数复制给目标网络
        self.target_net.load_state_dict(self.online_net.state_dict())


# ==========================================
# 5. 命令行训练、保存和独立评估
# ==========================================
def save_learning_curve(rewards, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = 20
    smoothed = [np.mean(rewards[max(0, i - window + 1):i + 1])
                for i in range(len(rewards))]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.3, color="gray", label="Episode return")
    ax.plot(smoothed, label=f"Moving average (window={window})")
    ax.set(xlabel="Episode", ylabel="Total reward", title="Dueling Double DQN / LunarLander-v3")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate(agent, episodes, seed, max_steps, render=False):
    env = gym.make("LunarLander-v3", max_episode_steps=max_steps,
                   render_mode="human" if render else None)
    scores = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            score = 0.0
            while True:
                action = agent.choose_action(state, evaluate=True)
                state, reward, terminated, truncated, _ = env.step(action)
                score += reward
                if terminated or truncated:
                    break
            scores.append(score)
            print(f"Eval {episode + 1}: return={score:.2f}", flush=True)
    finally:
        env.close()
    return scores


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate Dueling Double DQN on LunarLander-v3")
    parser.add_argument("--episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/default"))
    parser.add_argument("--checkpoint", type=Path, help="Online network weights for --eval-only")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="Open a window during evaluation")
    args = parser.parse_args()
    if min(args.episodes, args.eval_episodes, args.max_steps, args.threads) < 1:
        parser.error("episodes, eval-episodes, max-steps and threads must be positive")
    if args.eval_only and args.checkpoint is None:
        parser.error("--eval-only requires --checkpoint")
    if args.checkpoint is not None and not args.eval_only:
        parser.error("--checkpoint is evaluation-only, not a training resume option")

    global device
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 独立文件避免评估覆盖训练配置；相同output-dir重复训练会覆盖，须用不同目录。
    config_file = "eval_config.json" if args.eval_only else "config.json"
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"gymnasium": gym.__version__, "torch": str(torch.__version__),
                   "numpy": np.__version__, "gamma": GAMMA, "lr": LR,
                   "batch_size": BATCH_SIZE, "memory_capacity": MEMORY_CAPACITY,
                   "target_update_steps": TARGET_UPDATE_STEPS,
                   "epsilon_decay": EPSILON_DECAY, "min_epsilon": MIN_EPSILON})
    (args.output_dir / config_file).write_text(json.dumps(config, indent=2) + "\n")
    agent = Agent(8, 4)
    rewards = []
    global_step = updates = target_syncs = 0
    if args.eval_only:
        agent.online_net.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    else:
        env = gym.make("LunarLander-v3", max_episode_steps=args.max_steps)
        env.action_space.seed(args.seed)
        try:
            with (args.output_dir / "training.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["episode", "return", "steps", "global_step", "epsilon", "mean_loss"])
                for episode in range(args.episodes):
                    state, _ = env.reset(seed=args.seed if episode == 0 else None)
                    score = 0.0
                    losses = []
                    for step in range(args.max_steps):
                        action = agent.choose_action(state)
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        # 外部时间截断结束本局，但不清零TD自举；真正终止才清零。
                        agent.memory.push(state, action, reward, next_state, terminated)
                        loss = agent.update()
                        if loss is not None:
                            losses.append(loss)
                            updates += 1
                        global_step += 1
                        if global_step % TARGET_UPDATE_STEPS == 0:
                            agent.update_target_network()
                            target_syncs += 1
                        state = next_state
                        score += reward
                        if terminated or truncated:
                            break
                    rewards.append(score)
                    used_epsilon = agent.epsilon
                    agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)
                    writer.writerow([episode + 1, score, step + 1, global_step,
                                     used_epsilon, np.mean(losses) if losses else ""])
                    handle.flush()
                    print(f"Episode {episode + 1}: return={score:.2f}, steps={step + 1}, "
                          f"epsilon={used_epsilon:.3f}, updates={updates}", flush=True)
                    if len(rewards) >= 100 and np.mean(rewards[-100:]) >= 200:
                        print("Training rolling mean reached 200; evaluating separately.", flush=True)
                        break
        finally:
            env.close()
        torch.save(agent.online_net.state_dict(), args.output_dir / "model.pt")
        save_learning_curve(rewards, args.output_dir / "learning_curve.png")

    # 新环境、不同种子、关闭探索；不更新参数。
    eval_seed = args.seed + 100000
    scores = evaluate(agent, args.eval_episodes, eval_seed, args.max_steps, args.render)
    metrics = {"evaluation_returns": scores, "evaluation_mean": float(np.mean(scores)),
               "evaluation_std": float(np.std(scores)), "evaluation_seed_start": eval_seed,
               "evaluation_episodes": len(scores), "max_episode_steps": args.max_steps}
    if not args.eval_only:
        metrics.update({"training_episodes": len(rewards), "environment_steps": global_step,
                        "gradient_updates": updates, "target_syncs": target_syncs,
                        "training_last100_mean": float(np.mean(rewards[-100:]))})
    filename = "evaluation.json" if args.eval_only else "metrics.json"
    (args.output_dir / filename).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
