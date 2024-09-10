import numpy as np
def n_checkpoint_sample(intermediate_goals, n=2, cut = 0):
    # Step 1: Get intermediate goals using cumulative distance sampling

    # Step 2: Apply the cut to reduce the range
    cut = min(cut, len(intermediate_goals) - 1)  # Use self.cut as the cut value
    intermediate_goals = intermediate_goals[cut:]  # Cut from the start

    # Step 3: Calculate the step size for sampling
    idx_distance = max(1, len(intermediate_goals) // n)

    # Step 4: Ensure that at least the last element is included
    if n >= len(intermediate_goals):
        sampled_goals = intermediate_goals  # Take all if n is greater than remaining goals
    else:
        # Reverse sampling logic with gradual diminishing
        sampled_goals = [intermediate_goals[-1]]  # Start with the last element
        
        current_idx = len(intermediate_goals) - 1  # Start from the end
        while len(sampled_goals) < n:
            # Gradually decrease the step size
            idx_distance = max(1, current_idx // (n - len(sampled_goals)))
            current_idx = max(0, current_idx - idx_distance)
            sampled_goals.append(intermediate_goals[current_idx])

        # Reverse to maintain correct order
        sampled_goals.reverse()

    return np.array(sampled_goals)


for list_len in range(1,10):
    for cut in range(12):

        intermediate_goals = np.arange(list_len)
        print(f"{list_len},\t{cut},\t{n_checkpoint_sample(intermediate_goals,cut = cut, n =3)}")

