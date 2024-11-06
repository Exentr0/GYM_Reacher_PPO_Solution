import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std


class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.value(x)


class PPOAgent:
    def __init__(self, env, policy_lr=3e-4, value_lr=1e-3, gamma=0.99, epsilon=0.2, k_epochs=10):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.k_epochs = k_epochs

        obs_space = env.observation_space.shape[0]
        act_space = env.action_space.shape[0]

        self.policy = PolicyNetwork(obs_space, act_space)
        self.critic = CriticNetwork(obs_space)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=value_lr)

    def get_action_and_value(self, state):
        state = torch.FloatTensor(state)
        mean, std = self.policy(state)
        dist = MultivariateNormal(mean, torch.diag(std))
        action = dist.sample()
        value = self.critic(state)
        return action.numpy(), value, dist.log_prob(action)

    def compute_returns(self, rewards, values, dones):
        returns = []
        G = 0
        for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            G = reward + (1 - done) * self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)

    def update(self, states, actions, log_probs, rewards, values, dones):
        returns = self.compute_returns(rewards, values, dones)
        advantages = returns - torch.FloatTensor(values)
        total_policy_loss = 0
        total_value_loss = 0

        for _ in range(self.k_epochs):
            mean, std = self.policy(torch.FloatTensor(states))
            dist = MultivariateNormal(mean, torch.diag(std))
            new_log_probs = dist.log_prob(torch.FloatTensor(actions))
            ratio = torch.exp(new_log_probs - torch.FloatTensor(log_probs))

            # Policy Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            total_policy_loss += policy_loss.item()

            # Value Loss
            values_pred = self.critic(torch.FloatTensor(states))
            value_loss = ((returns - values_pred) ** 2).mean()
            total_value_loss += value_loss.item()

            # Update the Policy Network
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Update the Critic Network
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        # Calculate and print average losses
        avg_policy_loss = total_policy_loss / self.k_epochs
        avg_value_loss = total_value_loss / self.k_epochs
        print(f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")

    def train(self, episodes=1000000, max_timesteps=100):
        for episode in range(episodes):
            state = self.env.reset()[0]
            rewards, states, actions, log_probs, values, dones = [], [], [], [], [], []

            for t in range(max_timesteps):
                action, value, log_prob = self.get_action_and_value(state)
                next_state, reward, done, _, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value.item())
                rewards.append(reward)
                dones.append(done)

                state = next_state
                if done:
                    break

            self.update(states, actions, log_probs, rewards, values, dones)
            print(f"Episode {episode} finished")

    def evaluate(self, num_episodes=5):
        """Evaluates the trained agent with rendering."""
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            done = False
            total_reward = 0

            while not done:
                self.env.render()

                # Select action using the trained policy
                state = torch.FloatTensor(state)
                mean, _ = self.policy(state)
                action = mean.detach().numpy()

                # Step through the environment using the action
                state, reward, done, _, _ = self.env.step(action)
                total_reward += reward

            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

        self.env.close()


env = gym.make("Reacher-v4")

# Create and train the PPO agent
agent = PPOAgent(env)
agent.train()

# Recreate the environment with rendering for evaluation
env = gym.make("Reacher-v4", render_mode='human')
agent.env = env

# Evaluate the agent
agent.evaluate(num_episodes=5)
