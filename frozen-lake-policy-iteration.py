import numpy as np
import gym

DISCOUNT_RATE = 1.0
MAX_VALUE_ITERATIONS = 1000
MAX_POLICY_ITERATIONS = 10000
VALUE_ITERATION_THRESHOLD = 1e-10
MAX_STEPS = 100

def main():
    env = gym.make('FrozenLake-v0')
    optimal_policy = policy_iteration(env)

    # Use the optimal policy to play
    state = env.reset()
    env.render()
    total_reward = 0
    for _ in range(MAX_STEPS):
        action = optimal_policy[state]
        state, reward, is_done, _ = env.step(action)
        env.render()
        total_reward += reward
        if is_done:
            print('We finished!')
            break

    print(f'Total Reward: {total_reward}')

def compute_value_table(env, policy):
    value_table = np.zeros(env.observation_space.n)
    for _ in range(MAX_VALUE_ITERATIONS):
        old_value_table = np.copy(value_table)
        # for each state in the environment, select the action according to the policy and compute the value table
        for state_index in range(env.observation_space.n):
            action_index = policy[state_index]
            # Iterate over next states and also get reward and transition probability
            # Unecessary variable q_value, but it helps illustrate the math.
            q_value = 0
            for transition_prob, next_state_index, reward, _ in env.env.P[state_index][action_index]:
                 q_value += transition_prob*(reward + DISCOUNT_RATE*old_value_table[next_state_index])
            value_table[state_index] = q_value

        if np.sum(np.fabs(value_table - old_value_table)) <= VALUE_ITERATION_THRESHOLD:
            break

    return value_table

def extract_policy(env, value_table):
    policy = np.zeros(env.observation_space.n, dtype=int)
    for state_index in range(env.observation_space.n):
        q_values = np.zeros(env.action_space.n)
        # compute Q value for all actions in the state
        for action_index in range(env.action_space.n):
            for transition_prob, next_state_index, reward, _ in env.env.P[state_index][action_index]:
                q_values[action_index] += transition_prob*(reward + DISCOUNT_RATE*value_table[next_state_index])

        # Select the action which has maximum Q value as an optimal action of the state
        policy[state_index] = np.argmax(q_values)

    return policy

def policy_iteration(env):
    policy = np.zeros(env.observation_space.n, dtype=int)
    for iter_index in range(MAX_POLICY_ITERATIONS):
        value_table = compute_value_table(env, policy)
        new_policy = extract_policy(env, value_table)
        # Then we check whether we have reached convergence i.e whether we found the optimal
        # policy by comparing policy and new_policy if it same we will break the iteration
        # else we update policy with new_policy
        if np.array_equal(new_policy, policy):
            print(f'Policy Iteration converged at step {iter_index + 1}.')
            break

        policy = new_policy

    return policy

if __name__ == '__main__': main()
