import gym

import numpy as np

import torch

class PolicyGradientAgent(object):

    def __init__(self, env, policy, gamma=0.99, lr=0.001):

        self.env = env

        self.policy = policy

        self.gamma = gamma

        self.lr = lr

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self.log_probs = []

        self.rewards = []

    def train(self, num_episodes):

        for episode in range(num_episodes):

            # Reset the environment and get the initial observation.

            s = self.env.reset()

            # Initialize the episode rewards.

            episode_reward = 0

            # Loop until the episode is over.

            while True:

                # Choose an action according to the policy.

                a = self.policy(s)

                # Take the action and get the next observation and reward.

                s_prime, r, done, _ = self.env.step(a)

                # Store the log probability of the action and the reward.

                self.log_probs.append(self.policy.log_prob(s, a))

                self.rewards.append(r)

                # Update the policy.

                self.optimizer.zero_grad()

                loss = -torch.sum(self.log_probs * torch.stack(self.rewards).unsqueeze(1))

                loss.backward()

                self.optimizer.step()

                # If the episode is over, break.

                if done:
                  break

                # Update the state.

                s = s_prime

            # Print the episode reward.

            print("Episode reward:", episode_reward)

class Policy(torch.nn.Module):

    def __init__(self, obs_size, action_size):

        super(Policy, self).__init__()

        self.fc1 = torch.nn.Linear(obs_size, 128)

        self.fc2 = torch.nn.Linear(128, 64)

        self.fc3 = torch.nn.Linear(64, action_size)

    def forward(self, x):

        x = torch.relu(self.fc1(x))

        x = torch.relu(self.fc2(x))

        x = self.fc3(x)

        return x

def main():

    # Create the environment.

    env = gym.make("FrozenLake-v0")

    # Create the policy.

    policy = Policy(env.observation_space.n, env.action_space.n)

    # Create the agent.

    agent = PolicyGradientAgent(env, policy)

    # Train the agent.

    agent.train(1000)

    # Test the agent.

    for episode in range(10):

        s = env.reset()

        done = False
        while not done:

            a = agent.policy(s)

            s_prime, r, done, _ = env.step(a)

            env.render()

            print("State:", s)

            print("Action:", a)

            print("Reward:", r)

            s = s_prime
            import numpy as np

import matplotlib.pyplot as plt

# Plot the rewards over time.

rewards = []

for episode in range(1000):

    s = env.reset()

    done = False

    episode_reward = 0

    while not done:

        a = agent.policy(s)

        s_prime, r, done, _ = env.step(a)

        episode_reward += r

        rewards.append(episode_reward)

        s = s_prime

# Calculate the mean reward per episode.

mean_reward = np.mean(rewards)

# Plot the mean reward over time.

plt.plot(np.arange(len(rewards)), rewards)

plt.plot(np.arange(len(rewards)), [mean_reward] * len(rewards))

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.legend(["Rewards", "Mean Reward"])

plt.show()

# Save the policy.

torch.save(policy.state_dict(), "policy.pt")

# Load the policy.
for episode in range(10):

    s = env.reset()

    done = False

    while not done:

        a = policy(s)

        s_prime, r, done, _ = env.step(a)

        env.render()

        print("State:", s)

        print("Action:", a)

        print("Reward:", r)

        s = s_prime

# Plot the rewards over time.

rewards = []

for episode in range(1000):

    s = env.reset()

    done = False

    episode_reward = 0

    while not done:

        a = policy(s)

        s_prime, r, done, _ = env.step(a)

        episode_reward += r

        rewards.append(episode_reward)

        s = s_prime

# Calculate the mean reward per episode.

mean_reward = np.mean(rewards)

# Plot the mean reward over time.

plt.plot(np.arange(len(rewards)), rewards)

plt.plot(np.arange(len(rewards)), [mean_reward] * len(rewards))

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.legend(["Rewards", "Mean Reward"])

plt.show()

# Play the game using the loaded policy.

while True:

    s = env.reset()

    done = False
    while not done:

        a = policy(s)

        s_prime, r, done, _ = env.step(a)

        env.render()

        print("State:", s)

        print("Action:", a)

        print("Reward:", r)

        s = s_prime

        if done:

            break

# Add more functionality and end the program.

if mean_reward >= 100:

    print("You have successfully trained the agent to reach the goal!")

    break

else:

    print("You need to train the agent more.")
if mean_reward >= 100:

    print("You have successfully trained the agent to reach the goal!")

    break

else:

    print("You need to train the agent more.")

    

    # This code will end the program.

    import sys

    sys.exit()
    
