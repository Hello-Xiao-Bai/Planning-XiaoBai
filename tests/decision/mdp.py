import numpy as np

# 1. Define the state space.
states = np.arange(25)

# 2. Define the action space:
# 0 --> top, 1 --> right, 2 --> bottom, 3 --> left.
actions = np.arange(4)

# 3. Define the rewards.
rewards = np.ones(25) * -1.0
rewards[22] = 0.0

# 4. Define the discount factor.
gamma = 0.9


# 5. Define the transition matrix:
# input: current state, and action
# output: transition probability, next state, and reward.
def p_state_reward(state, action):
    # action: top --> 0
    if action == 0:
        if state in range(5):
            return (1.0, state, -1.0)
        else:
            return (1, state - 5, -1)

    # action: down --> 2
    if action == 2:
        if state in range(20, 25):
            return (1.0, state, -1)
        elif state == 17:
            return (1.0, state + 5, 0.0)
        else:
            return (1.0, state + 5, -1.0)

    # action: left --> 3
    if action == 3:
        if state in range(0, 25, 5):
            return (1.0, state, -1.0)
        elif state == 23:
            return (1.0, state - 1, 0.0)
        else:
            return (1.0, state - 1, -1.0)

    # action: right --> 1
    if action == 1:
        if state in range(4, 29, 5):
            return (1.0, state, -1.0)
        elif state == 21:
            return (1.0, state + 1, 0.0)
        else:
            return (1.0, state + 1, -1.0)


# 6. Solver with policy iteration.
# 6.1 Policy evaluation: calculate the state value in given strategy.
def compute_value_function(policy, gamma):
    # Set threshold.
    threshold = 1e-10

    # Initialize the state value.
    value_table = np.zeros(len(states))

    # Begin to iterate.
    while True:
        # Create the state value table in each iteration.
        update_value_table = np.copy(value_table)

        # Loop for each state.
        for state in states:
            # Get the action.
            action = policy[state]

            # Calculate the transition probability, next state, and reward.
            prob, next_state, reward = p_state_reward(state, action)

            # Calculate the state value.
            value_table[state] = reward + gamma * prob * update_value_table[next_state]

        # If the state_valueis converged, break the loop.
        if np.sum(np.fabs(value_table - update_value_table)) < threshold:
            break

    return value_table


# 6.2 Policy improvement: improve the current policy based on the state value.
def next_best_policy(value_table, gamma):
    # Initialize the policy.
    policy = np.zeros(len(states))

    # Loop for each state.
    for state in states:
        # Initialize the action value.
        action_table = np.zeros(len(actions))

        # Loop for each action.
        for action in actions:
            # Calculate the transition probability, next state, and reward.
            prob, next_state, reward = p_state_reward(state, action)

            # Calculate the action value.
            action_table[action] = prob * (reward + gamma * value_table[next_state])

        # Choose the best action.
        policy[state] = np.argmax(action_table)

    return policy


# 6.3 Construct policy iteration function.
def policy_iteration(random_policy, gamma, n):
    # Begin to iterate.
    for i in range(n):
        # Policy evaluation.
        new_value_function = compute_value_function(random_policy, gamma)

        # Policy improvement.
        new_policy = next_best_policy(new_value_function, gamma)

        # Judge the current policy.
        if np.all(random_policy == new_policy):
            print("End to iterate, and num is: %d" % (i + 1))
            break

        # Replace the optimal policy.
        random_policy = new_policy

    return new_policy


# 7. Solver with the value iteration.
def value_iteration(value_table, gamma, n):
    value_table = np.zeros(len(states))
    threshold = 1e-20
    policy = np.zeros(len(states))

    # Begin to iterate.
    for i in range(n):
        update_value_table = np.copy(value_table)

        # Loop for each state.
        for state in states:
            action_value = np.zeros(len(actions))

            # Loop for each action.
            for action in actions:

                # Calculate the transition probability, next state, and reward.
                trans_prob, next_state, reward = p_state_reward(state, action)

                # Calculate the action value.
                action_value[action] = (
                    reward + gamma * trans_prob * update_value_table[next_state]
                )

            # Update the state value table.
            value_table[state] = max(action_value)

            # Record the optimal policy
            policy[state] = np.argmax(action_value)

        ## End to iterate.
        if np.sum((np.fabs(update_value_table - value_table))) <= threshold:
            print("End to iterate, and num is: %d" % (i + 1))
            break

    return policy


kUseValueIteration = True


def main():
    # Set iteration num.
    n = 1000
    if kUseValueIteration:
        value_table = np.zeros(len(states))
        best_policy = policy_iteration(value_table, gamma, n)
    else:
        random_policy = 2 * np.ones(len(states))
        best_policy = policy_iteration(random_policy, gamma, n)

    print("best policy is: ", best_policy)

    # Find the optimal route.
    best_route = [0]
    next_state = 0
    while True:

        # Solve the next state to which the optimal action is transferred through the best strategy in the current state
        _, next_state, _ = p_state_reward(next_state, best_policy[next_state])

        # Add the next state to the best route list
        best_route.append(next_state)

        # Transfer to termination state, stop loop
        if next_state == 22:
            break

    print("The best route is: ", best_route)


if __name__ == "__main__":
    main()
