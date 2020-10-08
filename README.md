## What is Reinforcement Learning?
Reinforcement Learning is a subset of machine learning whose algorithms "learn correct behaviors" by seeking to maximize a numerical reward signal. We can draw an analogy from this numerical concept of a reward to the neural reward signals in a human or animal's brain that help assess whether a sensation or state is gratifying or not. In general, Reinforcement Learning algorithms seek to maximize this numerical reward as an animal would seek behaviors that lead to the brain generating the most pleasurable signals. 

Much like you and I use our past experiences to guide our decision-making, a Reinforcement Learning agent may also store models that retain information from previous experiences. These models help the agent plan for the future, influencing its behavior over time. Some of these models include its environment model, value estimates, and policy. 

Quotes contained within blockquotes throughout this write-up are from Sutton and Barto's, *Reinforcement Learning*.

### Environment Model
 > "By a model of the environment we mean anything that an agent can use to predict how the
environment will respond to its actions. Given a state and an action, a model produces a prediction of the resultant next state and next reward"

Imagine we are an agent learning how to play Tic-Tac-Toe optimally (in order to maximize our chances of winning). We can consider the current state during our turn to be the 3x3 board of blank spaces, X's, and O's. With a small, finite state space, it is possible for us to map out all of our possible moves (actions) at any given time and even attempt to predict the opponent's move in response to our own. But how would we know exactly how the opponent is going to play? If say, the opponent had a fixed strategy, then predicting future states following our next move would be feasible. We could bake this information about the opponent's strategy into our environment model to help us maximize our chance of winning.

Unfortunately, with most interesting problems, we don't have the luxury of complete information about the environment as we might in a Tic-Tac-Toe game against a predictable foe. Furthermore, the state space may be unknown or too large to be encoded and stored in memory. Nonetheless, we may seek to strike a balance between model-based methods and model-free methods by approximating a model of our environment in the neighborhood of states most relevant to our current state as opposed to keeping a complete model of all states in memory.

### Value Estimates
> Almost all reinforcement learning algorithms involve estimating value functions—functions
of states (or of state–action pairs) that estimate how good it is for the agent to be in a
given state (or how good it is to perform a given action in a given state). The notion
of “how good” here is defined in terms of future rewards that can be expected, or, to
be precise, in terms of expected return.

Let's revisit the Tic-Tac-Toe example from earlier. If we had some notion of the opponent's strategy or had simply made an initial guess as to how the opponent plays, we could consider the probability of winning from each state of the game board to be an estimate of the state’s value. We would have to use our assumption of the opponent's strategy to calculate the probability of winning in each state. The value function would then be the entire set of all state-value estimates. To maximize our chances of winning, the action to take would always be the action that leads to the state with the highest state value, assuming we know exactly how the opponent plays after each of our moves. Note that if our assumption of the opponent's strategy is not completely accurate or the strategy is simply not rigid, we may seek to adjust our value function over the course of the game to better shape our policy of action. Under these circumstances, we would consider our environment to be dynamic. 

### Policy
> Formally, a policy is a mapping from states to probabilities of selecting each possible
action.

The above definition applies to the probabilities that we, as a learning agent, will select our next possible action. So, if according our value function and the current state of the Tic-Tac-Toe board, putting an "O" in the bottom-left square of the board leads to the highest possible chance of winning, then our policy will give us a high probability, if not a 100% chance of selecting the bottom-left square as our next move.

You might be wondering why our policy doesn't simply prescribe a 100% chance of placing our "O" in the bottom-left corner if our value function tells us that this move leads to the highest probability of winning. Recall that sometimes, the environment is changing or is defined to be dynamic. In this example, a dynamic environment means we aren't completely sure about the opponent's strategy. Assume that our prediction of the opponent's strategy, which we are using to define our environment model and build our value function, is actually wrong. There could be a small probability that if we place our "O" in a different square, the opponent's real response (different from our current prediction of the opponent's response) actually leads to a higher chance of us winning.

In order to account for uncertainty in the environment, some Reinforcement Learning algorithms call for the agent to leave a probability of "exploration" in the policy, such that the agent chooses an action that may not be optimal according to the current value function. In the case that our exploratory action does not lead to a better outcome than the currently predicted "best" action, we would still use this experience to shape our current policy and adjust the value of the exploratory action according to how rewarding it really was. In the case that our exploratory action actually turns out better than we expect (and in extreme cases, even better than our currently predicted "best" action), we will increase the value of the exploratory action so we have a higher probability of choosing this action the next time we encounter the same state.

## Problem Descriptions
While the Tic-Tac-Toe example above is trivial, it can be illustrative of many Reinforcement Learning features pertinent to less trivial problems. The MountainCar and CartPole problems described below provide pedagogical Reinforcement Learning environments that can be tackled using a variety of different methods.  

### MountainCar
> A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

### Cartpole
> A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

## Preliminaries
Here are some definitions to give us a better idea of what's going on before we review some of the code:

- **Dynamic Programming (DP)** algorithms compute optimal policies (action plans) given a complete model of the environment. This assumption is unrealistic in the real world and too computationally expensive to be practical, but DP algorithms are still of theoretical importance. In fact, achieving the perfomance of a DP algorithm is what most Reinforcement Learning methods strive for.
- Given a policy that the agent is using to choose its actions, **policy evaluation** refers to the interative computation of a **value function** (described above)
- Given a value function and a current policy, **policy improvement** is the computation of an improved policy
- In general, we only have imperfect estimates of state values as the learning agent. Otherwise, there would be no value function and no policy to learn, and we would already have the optimal policy. **Bootstrapping** is the idea of updating state values on the basis of other estimated successor state values.
- A termintaing **episode** is defined as the period from which a task starts and ends, either with success or failure. In Reinforcement Learning tasks, we generally have a concept of time, where every time step implies that we take an action and transition to a new state, observing some negative, neutral, or positive reward associated with our action.
- Unlike DP methods, **Monte Carlo (MC)** methods don't assume complete knowledge of the enivronment. MC methods only need a model to generate sample transitions (rewards and next-states resulting from our agent's actions) as opposed to a complete distribution of transitions. 
- MC methods sample experiences from a sample model until the termintion of an episode. Only after an entire episode is sampled, does the MC algorithm adjust its value function and policy. Contrast this to one-step bootstrapping methods which adjust the value function and policy every step of an episode. 
- **Temporal Difference (TD)** methods combine the ideas of DP and MC methods by learning from sample experience without complete knowledge of the environment's dynamics/transition function. Unlike MC methods, TD methods don't aim to sample experience until termination of an episode. TD methods sample for n-steps from its sample model and then bootstrap the value of the state it has arrived in using successor state values. 
- Most real-world problems have state spaces that are much too large to be encoded in memory and subsequently, too large to allow us to compute relevant policies and value functions with limited computational resources. In our tasks, we will only be able to sample a small fraction of the possible states, so we need to be able to generalize from our limited experience, even when they are different from the states we will encounter in the future. **Function approximation** allows us to achieve this kind of generalization.
- **Policy-gradient** methods aim to compute or approximate an optimal policy directly, instead of attempting to approximate the value function. However, these methods may be more efficient if an approximate value function accompanies the policy.  

## The Setup
### MountainCar
At each time step, the state of our environment is defined by the following four observations:
```
Observation:
Type: Box(2)
Num    Observation               Min            Max
0      Car Position              -1.2           0.6
1      Car Velocity              -0.07          0.07

Actions:
Type: Discrete(3)
Num    Action
0      Accelerate to the Left
1      Don't accelerate
2      Accelerate to the Right

Reward:
Reward of 0 is awarded if the agent reached the flag (position = 0.5) on top of the mountain.
Reward of -1 is awarded if the position of the agent is less than 0.5
```

### Cartpole
```
Observation:
Type: Box(4)
Num     Observation               Min                     Max
0       Cart Position             -4.8                    4.8
1       Cart Velocity             -Inf                    Inf
2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
3       Pole Angular Velocity     -Inf                    Inf

Actions:
Type: Discrete(2)
Num   Action
0     Push cart to the left
1     Push cart to the right

Reward:
Reward is 1 for every step taken, including the termination step
```

## True Online Sarsa(λ) with Tile Coding
### Eligibility Traces
Like n-step bootstrapping, eligibility traces provide a mechanism to unify TD methods and MC methods. With eligibility traces, we can implement MC methods online (instead of incrementally by episode) and on continuing problems that don't need to terminate. With function approximation methods, we need a weight vector that produces a value estimate for a given state. Using eligibility traces implies keeping an eligibility vector that accompanies the weight vector. Each component of the eligibility vector is a trace component corresponding to each weight component of the weight vector. The trace component defines how long we should continue to "learn" in the corresponding component of the weight vector. For example, if a component of the weight vector is used to estimate the value of a state, we "activate" the corresponding eligibility trace in the eligibility vector to signal that we want to adjust this component of the weight vector until the eligibility trace decays to 0. A trace-decay parameter, which takes values in the range 0 to 1, determines how quickly the trace components in our eligibility vector decay.

### Tile Coding
In a problem with continuous state space, such as the MountainCar environment, we have an infinite number of states we may find ourselves in at any given point in time. We use a method called state aggregation to discretize the number of possible states so that we can define a policy that maps a state observation to a prescribed action or probability distribution over all actions. 

One method of multidimensional state aggregation is called tile coding. Imagine we have a 2d plane (two-dimensional state space). We're only interested in values between -1.2 and 0.6 for the x-axis. For the y-axis, we restrict ourselves to values between -0.07 and 0.07. There are an infinite combination of states, (x,y) pairs in this enclosed region, but we can discretize the states such that there are only a finite number. If you imagine we overlayed a tiling, or a grid of rectangles on top of our region of interest and grouped together states which fell into a single rectangle, we can turn our infinite state space into one with a finite number. Using a single tiling is called state aggregation. Tile coding calls for multiple overlapping tilings, each offset by a specified distance, to better discriminate the state that an agent is in. 

For the MountainCar problem, we have 2 dimensions of state -- position and velocity. If say, we had four tiles across the dimension representing position and four tiles across the dimension representing velocity, then each tiling would have 4 x 4 = 16 total tiles. Say we wanted 10 total tilings for finer grained discrimination in our state space. This would mean we have 16 x 10 = 160 total tiles, of which 10 will be activated at any given point in time (one per tiling). In the next section, we remind ourselves that we are actually estimating action values with the Sarsa(λ) algorithm, or the desirability of taking an action when we are in a given state. This means for each state, we have one of two possible actions to choose from. Our feature vector then has a total of 160 x 2 = 320 features.  

Given the desired bounds and desired tile width for each dimension of state, we create a function that generates **num_tilings** tilings. 

```py
# Returns num_tilings tilings
# Each tiling contains d arrays of intervals
# Each array of intervals defines the buckets/markers
# for each of the d dimensions of state
def create_tilings(self, state_low, state_high, num_tilings, tile_width):
    d = len(state_low)

    # Tile intervals/markers for each tiling
    # Each element tile_intervals will itself be list of d intervals
    # for each dimension of state
    tile_intervals = [[] for _ in range(num_tilings)]

    # Cumulative product of dimensions for determining tile number in tiling
    cum_prod_dims = [1]

    # Each tiling will cover all dimensions of the state exactly once
    for tiling_index in range(num_tilings):

        # Determine the tile intervals for each dimension of the state
        for dim in range(d):
            num_tiles = (
                int(np.ceil((state_high[dim] - state_low[dim]) / tile_width[dim]))
                + 1
            )

            offset = -(tiling_index / num_tilings) * tile_width[dim]
            low = state_low[dim] + offset
            high = low + tile_width[dim] * num_tiles

            # Determine the tile intervals
            # Each adjacent pair of interval markers will mark the start and end of each tile in that tiling
            # np.linspace(low, high, n) partitions range [low,high] into n-1 partitions and returns the markers
            # eg. np.linspace(0,1,3) -> [0, 0.5, 1]
            # Take [1: -1] for convenient bisection when searching for the interval
            markers = np.linspace(low, high, num_tiles + 1)[1:-1]

            tile_intervals[tiling_index].append(markers)

            # Last element is one_hot_encoding length for single tiling
            if tiling_index == 0:
                cum_prod_dims.append(num_tiles * cum_prod_dims[-1])

    return (np.array(tile_intervals), cum_prod_dims)
```

### Action Values
Action values are much like state values, except we are estimating the value or desirability of an action given our current state. Sarsa(λ) is a TD method for action values where λ represents the trace parameter. The pseudocode of True online Sarsa(λ) implemented in Sutton and Barto defines **w** to be the weight vector, **z** to be the eligibility trace vector, and **Q** the action-value function. In our implementation below, **s** denotes our current state, **a** the action prescribed by our current policy, and **x** the feature vector constructed using tile coding. Any variable with a subscript of `_prime` is our approximation of the variable at the next time step. 

```py
def SarsaLambda(
    env,  # openai gym environment
    gamma: float,  # discount factor
    lam: float,  # decay rate
    alpha: float,  # step size
    X: StateActionFeatureVectorWithTile,
    num_episode: int,
) -> np.array:
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
```

