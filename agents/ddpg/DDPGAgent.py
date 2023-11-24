import torch
from agents.RLAgent import RLAgent
from agents.ddpg.DDPGParams import DDPGParams
import torch.nn as nn


class DDPGAgent(RLAgent):
    def __init__(self, params: DDPGParams) -> None:
        super().__init__(params)

        self.encoder, self.actor, self.critic = params.build_model()
        self.target_encoder, self.target_actor, self.target_critic = params.build_model()

        self.exploration_sigma = params.exploration_sigma

        env = params.make_env()
        self.a_min = torch.tensor(env.action_space.low).to(self.device)
        self.a_max = torch.tensor(env.action_space.high).to(self.device)
        self.phi = params.phi

        self.optimiser = torch.optim.Adam(
            self.actor.parameters(),
            self.critic.parameters(),
            lr=params.lr,
        )

    @staticmethod
    def load(dir_path: str) -> "DDPGAgent":
        args = DDPGParams()
        args.load_json_file(f"{dir_path}/params.json")

        agent = DDPGAgent(args)
        agent.load_state_dict(torch.load(f"{dir_path}/model.pt"))

        return agent

    def get_action(self, state):
        with torch.no_grad():
            features = self.encoder(self.to_torch_batch(state))
            action = self.actor(features)
            noisy_action = action + \
                torch.randn_like(action) * self.exploration_sigma
            clipped_action = torch.clamp(noisy_action, self.a_min, self.a_max)
            return clipped_action.detach().cpu().numpy()[0]

    def train_model(self, transitions, episode):
        states, actions, rewards, next_states, dones = self.transition_to_torch_batch(
            transitions)

        encoded_features = self.encoder(states)
        N = len(states)
        # critic
        Q = self.critic(torch.cat([encoded_features, actions], dim=1))

        with torch.no_grad():
            encoded_features_ = self.target_encoder(next_states)
            actions_ = self.target_actor(encoded_features_)
            Q_ = self.target_critic(
                torch.cat([encoded_features_, actions_], dim=1))

        target = rewards + self.gamma * Q_.view(N) * (1 - dones)
        critic_loss = nn.functional.mse_loss(Q, target.view(N, 1))

        # actor
        pred_actions = self.actor(encoded_features)
        pred_values = self.critic(
            torch.cat([encoded_features, pred_actions], dim=1))
        actor_loss = - pred_values.mean()

        # update
        self.self.optimiser.zero_grad()

        total_loss = actor_loss + critic_loss
        total_loss.backward()

        self.optimiser.step()

        # update target neworks
        self._update_target_network(self.target_encoder, self.encoder)
        self._update_target_network(self.target_actor, self.actor)
        self._update_target_network(self.target_critic, self.critic)

        # log
        self.log("actor_loss", actor_loss, episode)
        self.log("critic_loss", critic_loss, episode)
        self.log("total_loss", actor_loss+critic_loss, episode)

    def _update_target_network(self, target_network, network):
        target_state_dict = target_network.state_dict()
        network_state_dict = network.state_dict()
        resulting_state_dict = {}
        for name, param in target_state_dict.items():
            resulting_state_dict[name] = self.phi * \
                param + (1 - self.phi) * network_state_dict[name]
        target_network.load_state_dict(resulting_state_dict)
