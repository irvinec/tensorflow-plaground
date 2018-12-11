import gym
import numpy as np

NUM_VALUE_ITERATIONS = 10000
STATE_VALUE_TABLE_DELTA_THRESHOLD_FOR_CONVERGENCE = 1e-20
DISCOUNT_FACTOR = 1.0

def main():
    env = gym.make('FrozenLake-v0')
    # Get the value function table for the optimal policy
    optimal_value_table = value_iteration(env)

    # Now that we have the optimal value function in terms of a table (value_table)
    # we need to extract the optimal policy so that we can choose the right action.
    optimal_policy = extract_policy(env, optimal_value_table)

    # Now play the game with the optimal policy
    total_reward = 0
    current_state = env.reset()
    env.render()
    for _ in range(100):
        best_action = optimal_policy[current_state]
        current_state, reward, is_done, _ = env.step(best_action)
        env.render()
        total_reward += reward
        if is_done:
            print('We reached a terminal state!')
            break

    print('The game is finished.')
    print(f'Total reward: {total_reward}')

def value_iteration(env):
    # Do value iteration algorithm
    # Psuedo-code:
    # Initialize V_0(s) to arbitrary values.
    # while V_i(s) not converged:
    #   for s in States:
    #       for a in Actions:
    #           Q(s,a) <- R(s,a) + DISCOUNT_FACTOR*sum(T(s,a,s')*V_i(s'))
    #       V_i+1(s) <- argmax(Q(s,a))

    # Initialize value table to all zeros.
    value_table = np.zeros(env.observation_space.n)

    for iter_index in range(NUM_VALUE_ITERATIONS):
        # Save the old value_table values so we can compare for convergence.
        old_value_table = np.copy(value_table)
        for state_index in range(env.observation_space.n):
            # Table of q values for this state
            q_values = np.zeros(env.action_space.n)
            for action_index in range(env.action_space.n):
                # NOTE: We pass in old_value_table to compute_q_value.
                # This reflects the Bellman update where we use V_i to update V_i+1
                q_values[action_index] = compute_q_value(env, old_value_table, state_index, action_index)
            value_table[state_index] = max(q_values)
        # We will check whether we have reached the convergence i.e whether the difference 
        # between our value table and updated value table is very small.
        # But how do we know it is very small?
        # We set some threshold and then we will see if the difference is less
        # than our threshold, if it is less, we break the loop and return the value function as optimal
        # value function.
        if np.sum(np.fabs(value_table - old_value_table)) <= STATE_VALUE_TABLE_DELTA_THRESHOLD_FOR_CONVERGENCE:
            print ('Value-iteration converged at iteration number: {}.'.format(iter_index + 1))
            break

    return value_table

def compute_q_value(env, value_table, state_index, action_index):
    # Compute the q_value the given state and action.
    # NOTE: this is a stochastic environment so we need to take into account the transtion probability.
    # R(s,a): The expected reward for taking action a when in state s
    # T(s,a,s'): The transition probability. The probability of ending up in state s'
    #   by taking action a in state s.
    # V(s): The "value" of being in state s.
    # s': Used to denote a new state reached by taking action a in state s
    # Q(s,a) = R(s,a) + DISCOUNT_FACTOR*sum_over_next_state(T(s,a,s')V(s'))
    # since expected reward, R(s,a) = sum_over_next_states(r(s,a,s')*T(s,a,s'))
    # where r(s,a,s') is the actual reward received when taking action a in state s
    # and getting to state s'
    # we can rewrite Q(s,a) to be
    # Q(s, a) = sum_over_next_states(T(s,a,s')*(r(s,a,s') + DISCOUNT_FACTOR*V(s'))

    # NOTE: env.P allows us to take a peak at the internal definitions of the environment.
    # This is cheating since normally our agent wouldn't have access to this information directly.
    # The agent would normally need to estimate this information from collected observations.
    # env.P essentially gives us a list of properties about all the possible states
    # that are reachable from a state by taking an action. old-state -> action -> new state.
    # Here we are iterating over all those entries for the given state and action.
    q_value = 0
    for transition_prob, next_state_index, reward, _ in env.env.P[state_index][action_index]:
        q_value += transition_prob*(reward + DISCOUNT_FACTOR*value_table[next_state_index])
    return q_value

def extract_policy(env, value_table):
    # Extract the policy as an array with indices being indexes of states.
    # policy[s] is the action that should be taken in state with index s
    # NOTE: We could have computed the policy as part of the value iteration,
    # but we wanted the algorithm to match as close as possible with typical value iteration.
    policy = np.zeros(env.observation_space.n, dtype=int)
    for state_index in range(env.observation_space.n):
        # Initialize the Q values for a state
        q_values = np.zeros(env.action_space.n)
        # Compute Q value for all actions in the state
        for action_index in range(env.action_space.n):
            q_values[action_index] = compute_q_value(env, value_table, state_index, action_index)
        # Select the action which has maximum Q value as an optimal action of the state
        policy[state_index] = np.argmax(q_values)

    return policy

if __name__ == "__main__":
    main()