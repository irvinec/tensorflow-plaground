import numpy as np
import gym

# Hyperparameters
NUM_EPISODES = 1000
MAX_STEPS = 99

LEARNING_RATE = 0.7
DISCOUNT_RATE = 0.6

MAX_EXPLORATION_RATE = 1.0
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY_RATE = 0.01

RENDER_TRAINING_EPISODES = False

def main():
    env = gym.make('Taxi-v2')
    q_table = train(env)
    play(env, q_table)
    env.close()

def train(env):
    env = gym.make('Taxi-v2')

    # Initialize Q table
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    q_table = np.zeros((num_states, num_actions))

    # Train the Q table
    for episode_index in range(NUM_EPISODES):
        # Decay the exploration rate
        # NOTE: that for the episode_index = 0
        # the exploration_rate is set to MAX_EXPLORATION_RATE as intended.
        exploration_rate = MIN_EXPLORATION_RATE + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE)*np.exp(-EXPLORATION_DECAY_RATE*episode_index)
        state = env.reset()
        if RENDER_TRAINING_EPISODES:
            env.render()
        for _ in range(MAX_STEPS):
            if np.random.uniform() <= exploration_rate:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            new_state, reward, is_done, _ = env.step(action)
            if RENDER_TRAINING_EPISODES:
                env.render()

            # Bellman equation
            # Q(s,a) <- Q(s,a) + alpha*[R(s,a) + gamma*max(Q(s',a')) - Q(s,a)]
            q_table[state, action] = q_table[state, action] + LEARNING_RATE*(reward + DISCOUNT_RATE*np.max(q_table[new_state, :]) - q_table[state, action])

            state = new_state
            if is_done:
                break

    return q_table

def play(env, q_table):
    # Play the game using our Q table to approximate the optimal policy
    state = env.reset()
    env.render()
    total_reward = 0
    for _ in range(MAX_STEPS):
        action = np.argmax(q_table[state, :])
        new_state, reward, is_done, _ = env.step(action)
        env.render()
        total_reward += reward
        if is_done:
            break
        state = new_state

    print(f'Total Reward: {total_reward}')

if __name__ == "__main__": main()