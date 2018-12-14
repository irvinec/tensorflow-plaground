import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from collections import Counter

ACTION_HIT = 1
ACTION_STAND = 0

NUM_EPISODES = 500000

def main():
    plt.style.use('ggplot')

    env = gym.make('Blackjack-v0')
    value_table = first_visit_monte_carlo_prediction(env, sample_policy)

    _, axes = plt.subplots(nrows=2, figsize=(5, 8), subplot_kw={'projection': '3d'})
    axes[0].set_title('value function with ace as 1')
    axes[1].set_title('value function with ace as 11')
    plot_blackjack(value_table, axes)

def first_visit_monte_carlo_prediction(env, policy):
    # http://incompleteideas.net/book/bookdraft2018jan1.pdf
    # Richard Sutton, Andrew Barto
    # RL: An Introduction
    # Section 5.1
    # This differs from 'Hands-On RL with Python' Sudharsan Ravichandaran
    # He uses an incremental update based on V' = V + (r - V)/n
    # This follows from formula for adding one more number to an average of n-1 numbers.
    value_table = defaultdict(float)
    returns = defaultdict(list)
    for _ in range(NUM_EPISODES):
        observations, _, rewards  = generate_episode(env, policy)
        sum_of_rewards = 0
        # Iterate backwards so that we can get the sum of rewards
        # after the first occurrence of observation.
        # (as described in Sutton and Barto)
        for step_index, observation in reversed(list(enumerate(observations))):
            sum_of_rewards += rewards[step_index]
            # If this is the first time we have seen this observation,
            # update the value table.
            # This makes it 'first visit Monte Carlo'
            if observation not in observations[:step_index]:
                returns[observation].append(sum_of_rewards)
                value_table[observation] = np.mean(returns[observation])

    return value_table

def generate_episode(env, policy):
    observations = []
    actions = []
    rewards = []
    observation = env.reset()
    done = False
    while not done:
        # Note we keep the first observation,
        # but don't store the last observation.
        # Consistent with S_0, A_0, R_1
        observations.append(observation)
        action = policy(observation)
        actions.append(action)
        observation, reward, done, _ = env.step(action)
        rewards.append(reward)

    return observations, actions, rewards

def sample_policy(observation):
    player_score, _, _ = observation
    return ACTION_STAND if player_score >= 20 else ACTION_HIT

def plot_blackjack(value_table, axes):
    player_sum = np.arange(12, 21 + 1)
    dealer_show = np.arange(1, 10 + 1)
    ace_as_11 = np.array([False, True])
    state_values = np.zeros((len(player_sum), len(dealer_show), len(ace_as_11)))
    for i, player in enumerate(player_sum):
        for j, dealer in enumerate(dealer_show):
            for k, ace in enumerate(ace_as_11):
                state_values[i, j, k] = value_table[player, dealer, ace]

    X, Y = np.meshgrid(player_sum, dealer_show)

    axes[0].plot_wireframe(X, Y, state_values[:, :, 0])
    axes[1].plot_wireframe(X, Y, state_values[:, :, 1])

    for axis in axes:
        axis.set_zlim(-1, 1)
        axis.set_ylabel('player sum')
        axis.set_xlabel('dealer showing')
        axis.set_zlabel('state-value')

    plt.show()

if __name__ == "__main__": main()