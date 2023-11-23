import torch
import torch.nn as nn
from agents.dqn.DeepQParams import DeepQParams


class DeepQNetwork(torch.nn.Module):
    """
    Network used by the DeepQAgent
    """

    def __init__(self, params: DeepQParams) -> None:
        super().__init__()

        torch.manual_seed(params.seed)

        self.model = params.build_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.lr)

    def forward(self, x):
        return self.model(x)

    def train_step(self, transitions, gamma, tdqn):
        states, actions, rewards, next_states, dones = transitions
        N = len(states)
        q = self(states)

        # select q values for the actions taken
        q = q.gather(1, actions.view(N, 1).long())
        q = q.view(N)

        with torch.no_grad():
            next_q = tdqn(next_states).max(dim=1)[0] * (1 - dones)

        target = rewards + gamma * next_q
        loss = nn.functional.huber_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
