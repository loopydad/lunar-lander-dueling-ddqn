"""关键算法语义核对；不测试是否学会着陆。"""
import unittest

import numpy as np
import torch

from lunar_lander import Agent, BATCH_SIZE, DuelingDQN, GAMMA


class DQNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def test_dueling_baseline_and_action_differences(self):
        net = DuelingDQN(8, 4)
        states = torch.randn(3, 8)
        features = net.feature_layer(states)
        q = net(states)
        value = net.value_stream(features)
        advantage = net.advantage_stream(features)
        torch.testing.assert_close(q.mean(dim=1, keepdim=True), value)
        torch.testing.assert_close(q[:, 0] - q[:, 2], advantage[:, 0] - advantage[:, 2])

    def test_warmup_update_freezes_target_and_hard_sync(self):
        agent = Agent(8, 4)
        state = np.ones(8, dtype=np.float32)
        for _ in range(BATCH_SIZE - 1):
            agent.memory.push(state, 2, 1.0, state, False)
        self.assertIsNone(agent.update())
        before = [p.detach().clone() for p in agent.online_net.parameters()]
        target_before = [p.detach().clone() for p in agent.target_net.parameters()]
        agent.memory.push(state, 2, 1.0, state, False)
        loss = agent.update()
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(p, old) for p, old in zip(agent.online_net.parameters(), before)))
        for p, old in zip(agent.target_net.parameters(), target_before):
            torch.testing.assert_close(p, old)
            self.assertIsNone(p.grad)
        agent.update_target_network()
        for online, target in zip(agent.online_net.parameters(), agent.target_net.parameters()):
            torch.testing.assert_close(online, target)

    def check_target(self, terminated):
        agent = Agent(8, 4)
        with torch.no_grad():
            for net in (agent.online_net, agent.target_net):
                for param in net.parameters():
                    param.zero_()
            # 在线选动作2；目标网络自己的最大动作是1，必须仍读取动作2。
            agent.online_net.advantage_stream[-1].bias.copy_(torch.tensor([0., 0., 2., 0.]))
            agent.target_net.advantage_stream[-1].bias.copy_(torch.tensor([1., 4., 3., 2.]))
        state = np.zeros(8, dtype=np.float32)
        for _ in range(BATCH_SIZE):
            agent.memory.push(state, 2, 1.0, state, terminated)
        # 当前Q(动作2)=1.5；目标Q(动作2)=0.5，不是max目标Q=1.5。
        expected_target = 1.0 if terminated else 1.0 + GAMMA * 0.5
        self.assertAlmostEqual(agent.update(), (1.5 - expected_target) ** 2, places=5)

    def test_double_target_uses_online_action(self):
        self.check_target(False)

    def test_terminal_transition_has_no_bootstrap(self):
        self.check_target(True)


if __name__ == '__main__':
    unittest.main()
