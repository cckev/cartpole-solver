import numpy as np

class StateActionFeatureVectorWithTile:
    def __init__(
        self,
        state_low: np.array,
        state_high: np.array,
        num_actions: int,
        num_tilings: int,
        tile_width: np.array,
    ):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """

        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions

        # create tilings and get dimension of the one-hot feature vector for state only
        (
            self.tile_intervals,
            self.cum_prod_dims,
            self.tiles_per_dim,  # DEBUGGING
            self.offset_per_dim,  # DEBUGGING
            self.bounds_per_dim,  # DEBUGGING
        ) = self.create_tilings(state_low, state_high, num_tilings, tile_width)
        self.num_tilings = num_tilings

    # Returns num_tilings tilings
    # Each tiling contains d arrays of intervals
    # Each array of intervals defines the buckets/markers
    # for each of the d dimensions of state
    def create_tilings(self, state_low, state_high, num_tilings, tile_width):
        assert (
            len(state_low) == len(state_high) == len(tile_width)
        ), "Dimensions in parameters inconsistent"
        assert len(state_low) > 0, "Must have at least one dimension of state"
        assert all(
            np.less(state_low, state_high)
        ), "state_low must be less than state_high"

        d = len(state_low)

        tiles_per_dim = []
        offset_per_dim = []
        bounds_per_dim = []

        # Tile intervals/markers for each tiling
        # Each element tile_intervals will itself be list of d intervals
        # for each dimension of state
        tile_intervals = [[] for _ in range(num_tilings)]

        # Cumulative product of dimensions for determining tile number in tiling
        cum_prod_dims = [1]

        # Each tiling will cover all dimensions of the state exactly once
        for tiling_index in range(num_tilings):

            tiles_per_dim.append([])
            offset_per_dim.append([])

            # Determine the tile intervals for each dimension of the state
            for dim in range(d):
                num_tiles = (
                    int(np.ceil((state_high[dim] - state_low[dim]) / tile_width[dim]))
                    + 1
                )

                offset = -(tiling_index / num_tilings) * tile_width[dim]
                low = state_low[dim] + offset
                high = low + tile_width[dim] * num_tiles

                # # DEBUGGING
                if tiling_index == 0:
                    bounds_per_dim.append(
                        [state_low[dim], state_low[dim] + tile_width[dim] * num_tiles]
                    )

                # Determine the tile intervals
                # Each adjacent pair of interval markers will mark the start and end of each tile in that tiling
                # np.linspace(low, high, n) partitions range [low,high] into n-1 partitions and returns the markers
                # eg. np.linspace(0,1,3) -> [0, 0.5, 1]
                # Take [1: -1] for convenient bisection when searching for the interval
                markers = np.linspace(low, high, num_tiles + 1)[1:-1]

                tile_intervals[tiling_index].append(markers)

                tiles_per_dim[-1].append(num_tiles)
                offset_per_dim[-1].append(offset)

                # Last element is one_hot_encoding length for single tiling
                if tiling_index == 0:
                    cum_prod_dims.append(num_tiles * cum_prod_dims[-1])

        return (
            np.array(tile_intervals),
            cum_prod_dims,
            tiles_per_dim,
            offset_per_dim,
            bounds_per_dim,
        )

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.num_actions * self.num_tilings * self.cum_prod_dims[-1]

    def __call__(self, s, done, a) -> np.array:
        """
        Function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        if done:
            return np.zeros((self.feature_vector_len(),), dtype=int)

        tile_numbers = np.zeros((self.feature_vector_len(),), dtype=int)
        # grid_encoding = []

        tiling_offset = a * self.cum_prod_dims[-1] * self.num_tilings

        # For every tiling, loop through all dimensions of the state
        # to find which tile the current dimension of state falls in
        for tiling_index in range(self.num_tilings):
            # grid_encoding.append([])

            tile_number = 0

            for dim, value in enumerate(s):
                assert (
                    self.state_low[dim] <= value <= self.state_high[dim]
                ), "State dimension {dim} out of bounds"

                # search tile intervals for state
                tile_index = np.searchsorted(
                    self.tile_intervals[tiling_index][dim], value
                )

                # grid_encoding[-1].append(tile_index)

                # update tile number
                tile_number += tile_index * self.cum_prod_dims[dim]

            tile_numbers[tile_number + tiling_offset] = 1

            # update offset for next tiling
            tiling_offset += self.cum_prod_dims[-1]

        return tile_numbers


def SarsaLambda(
    env,  # openai gym environment
    gamma: float,  # discount factor
    lam: float,  # decay rate
    alpha: float,  # step size
    X: StateActionFeatureVectorWithTile,
    num_episode: int,
) -> np.array:
    """
    True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s, done, w, epsilon=0.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]
        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))
    epsilon = 0.01

    for episode in range(1, num_episode + 1):
        if episode % 100 == 0:
            print(f"SarsaLambda episode: {episode}")
        s, done = env.reset(), False
        a = epsilon_greedy_policy(s, done, w, epsilon)
        x = X(s, done, a)
        z = np.zeros((X.feature_vector_len()))
        Q_old = np.zeros((X.feature_vector_len()))
        while not done:
            s, r, done, _ = env.step(a)
            a = epsilon_greedy_policy(s, done, w, epsilon)
            x_prime = X(s, done, a)
            Q = np.dot(w, x)
            Q_prime = np.dot(w, x_prime)
            delta = r + gamma * Q_prime - Q
            z = gamma * lam * z + (1 - alpha * lam * gamma * np.dot(z, x)) * x
            w = w + alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_prime
            x = x_prime
    return w

