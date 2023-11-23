import gym
import numpy as np
import torch
from agents.dqn.DeepQNetwork import DeepQNetwork
from agents.dqn.DeepQParams import DeepQParams
from agents.RLAgent import RLAgent
from agents.ReplayBuffer import ReplayBatch


class DeepQAgent(RLAgent):
    """
    A Deep Q Learning agent
    """

    def __init__(self, params: DeepQParams) -> None:
        super().__init__(params)
        self.target_update_steps = params.target_update_steps
        self.dqn = DeepQNetwork(params)
        self.tdqn = DeepQNetwork(params)
        self.dqn.to(params.device)
        self.tdqn.to(params.device)

    def get_action(self, state: gym.Space) -> gym.Space:
        with torch.no_grad():
            q = self.dqn(self.to_torch_batch(state))[0].detach().cpu().numpy()

            qmax = q.max()
            best = [a for a in range(self.env.action_space.n)
                    if np.allclose(qmax, q[a])]
            action = self.random_state.choice(best)
            return action

    def train_model(self, transitions: ReplayBatch, episode: int) -> None:
        loss = self.dqn.train_step(self.transition_to_torch_batch(
            transitions), self.gamma, self.tdqn)
        self.log("loss", loss, episode)

    def end_train_episode(self, episode: int, total_steps: int) -> None:
        if total_steps % self.sync_target_every_n_episodes == 0:
            self.tdqn.load_state_dict(self.dqn.state_dict())

    @staticmethod
    def load(dir_path: str) -> "DeepQAgent":
        args = DeepQParams()
        args.load_json_file(f"{dir_path}/params.json")

        agent = DeepQAgent(args)
        agent.load_state_dict(torch.load(f"{dir_path}/model.pt"))

        return agent
