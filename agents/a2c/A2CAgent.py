from agents.a2c.A2CParams import A2CParams
from agents.RLAgent import RLAgent


import torch
import torch.nn as nn


class A2CAgent(RLAgent):

    def __init__(self, params: A2CParams) -> None:
        super().__init__(params)
        self.shared_conv, self.actor, self.critic = params.build_model()
        self.actor_optimizer = torch.optim.Adam(
            list(self.shared_conv.parameters()) +
            list(self.actor.parameters()),
            lr=params.lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.shared_conv.parameters()) +
            list(self.critic.parameters()),
            lr=params.lr,
        )

    @staticmethod
    def load(dir_path: str) -> "A2CAgent":
        args = A2CParams()
        args.load_json_file(f"{dir_path}/params.json")

        agent = A2CAgent(args)
        agent.load_state_dict(torch.load(f"{dir_path}/model.pt"))

        return agent

    def get_action(self, state):
        with torch.no_grad():
            conv_output = self.shared_conv(self.to_torch_batch(state))
            logits = self.actor(conv_output)
            return torch.distributions.Categorical(logits=logits).sample().item()

    def train_model(self, transitions, episode):

        states, actions, rewards, next_states, dones = self.transition_to_torch_batch(
            transitions)
        N = len(states)

        conv_output = self.shared_conv(states)

        # critic
        Vs = self.critic(conv_output)
        with torch.no_grad():
            Vs_ = self.critic(self.shared_conv(next_states))

        advantage = rewards + self.gamma * Vs_.view(N) * (1 - dones)
        critic_loss = nn.functional.mse_loss(Vs, advantage.view(N, 1))

        # actor
        log_probs = self.actor(conv_output).log_softmax(dim=1)
        log_probs = log_probs[:, actions.long()]
        actor_loss = -(log_probs * advantage.detach()).mean()

        # update
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        critic_loss.backward(retain_graph=True)
        actor_loss.backward()

        self.critic_optimizer.step()
        self.actor_optimizer.step()

        # log
        self.log("actor_loss", actor_loss, episode)
        self.log("critic_loss", critic_loss, episode)
        self.log("total_loss", actor_loss+critic_loss, episode)
