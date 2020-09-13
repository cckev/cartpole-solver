from typing import Iterable
import numpy as np
import tensorflow as tf

class PiApproximationWithNN:
    def __init__(self, state_dims, num_actions, alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        self.var_scope = "PA"
        self.alpha = min(3e-6, alpha)
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.steps = 0
        self.init_model()
        self.sess, self.gradient_adjustments = self.init_session()

    def init_model(self):
        with tf.variable_scope(self.var_scope):
            self.weights = {
                "h1": tf.Variable(
                    tf.random.normal([self.state_dims, 32])
                ),  # state_dims inputs, 32 nodes
                "h2": tf.Variable(tf.random.normal([32, 32])),  # 32 nodes
                "out": tf.Variable(
                    tf.random.normal([32, self.num_actions])
                ),  # num_actions output
            }

            self.state = tf.compat.v1.placeholder(
                tf.float32, [None, self.state_dims], name="state"
            )
            self.action = tf.compat.v1.placeholder(tf.int32, [None], name="action")
            self.delta = tf.compat.v1.placeholder(tf.float32, [None], name="delta")

            # hidden layers
            layer_1 = tf.matmul(self.state, self.weights["h1"])
            layer_1 = tf.nn.relu(layer_1)
            layer_2 = tf.matmul(layer_1, self.weights["h2"])
            layer_2 = tf.nn.relu(layer_2)

            # output layer
            self.logits = tf.matmul(layer_2, self.weights["out"])

            self.pi = tf.nn.softmax(self.logits)
            self.selected_action_prob = tf.gather_nd(
                self.pi,
                tf.stack([tf.range(tf.shape(self.action)[0]), self.action], axis=1),
            )

            self.loss = tf.reduce_mean(
                -tf.math.log(tf.clip_by_value(self.selected_action_prob, 1e-10, 1))
                * self.delta
            )

            self.trainable_vars = tf.compat.v1.trainable_variables(self.var_scope)

            # contains gradients for all trainable variables
            self.gradient_placeholders = []
            for _ in range(len(self.trainable_vars)):
                self.gradient_placeholders.append(tf.compat.v1.placeholder(tf.float32))

            self.gradients = tf.gradients(self.loss, self.trainable_vars)

            self.optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.alpha, beta1=0.9, beta2=0.999
            )

            self.train_op = self.optimizer.apply_gradients(
                zip(self.gradient_placeholders, self.trainable_vars)
            )

    def init_session(self):
        sess = tf.compat.v1.Session()
        sess.run(tf.global_variables_initializer())
        gradient_adjustments = sess.run(
            tf.compat.v1.trainable_variables(self.var_scope)
        )
        for i, grad in enumerate(gradient_adjustments):
            gradient_adjustments[i] = grad * 0
        return sess, gradient_adjustments

    def __call__(self, s) -> int:
        s = np.reshape(s, (1, self.state_dims))
        probs = self.sess.run(self.pi, feed_dict={self.state: s})[0]
        action = np.random.choice(range(self.num_actions), p=probs)

        return action

    def set_gradient_adjustments(self, s, a, gamma_t, delta):
        gradient_adjustments = self.get_grads(s, a, delta)
        for i, grad in enumerate(gradient_adjustments):
            self.gradient_adjustments[i] += grad * gamma_t

    def clear_gradient_adjustments(self):
        for i, grad in enumerate(self.gradient_adjustments):
            self.gradient_adjustments[i] += grad * 0

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        s = np.reshape(s, (1, self.state_dims))
        a = np.reshape(a, (1,))
        gamma_t = np.reshape(gamma_t, (1,))
        delta = np.reshape(delta, (1,))

        self.set_gradient_adjustments(s, a, gamma_t, delta)

        # set feed for adjusting gradient for each training var
        feed = dict(zip(self.gradient_placeholders, self.gradient_adjustments))
        self.sess.run(self.train_op, feed_dict=feed)

        self.clear_gradient_adjustments()

        # if self.steps % 1000 == 0:
        #     print(f"PA step: {self.steps}")

        self.steps += 1

    def get_vars(self):
        return self.sess.run(tf.compat.v1.trainable_variables())

    def get_grads(self, s, a, delta):
        feed = {self.state: s, self.action: a, self.delta: delta}
        return self.sess.run(self.gradients, feed_dict=feed)


class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """

    def __init__(self, b):
        self.b = b

    def __call__(self, s) -> float:
        return self.b

    def update(self, s, G):
        pass


class VApproximationWithNN(Baseline):
    def __init__(self, state_dims, alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.var_scope = "VA"
        self.alpha = min(3e-6, alpha)
        self.state_dims = state_dims
        self.steps = 0
        self.init_model()
        self.sess, self.gradient_adjustments = self.init_session()

    def init_model(self):
        with tf.variable_scope(self.var_scope):
            self.weights = {
                "h1": tf.Variable(
                    tf.random.normal([self.state_dims, 32])
                ),  # state_dims inputs, 32 nodes
                "h2": tf.Variable(tf.random.normal([32, 32])),  # 32 nodes
                "out": tf.Variable(tf.random.normal([32, 1])),
            }

            self.state = tf.compat.v1.placeholder(
                tf.float32, [None, self.state_dims], name="state"
            )
            self.target_return = tf.compat.v1.placeholder(
                tf.float32, [None, 1], name="return"
            )

            # hidden layers
            layer_1 = tf.matmul(self.state, self.weights["h1"])
            layer_1 = tf.nn.relu(layer_1)
            layer_2 = tf.matmul(layer_1, self.weights["h2"])
            layer_2 = tf.nn.relu(layer_2)

            # output layer
            self.out = tf.matmul(layer_2, self.weights["out"])

            self.loss = tf.reduce_mean(tf.square(self.target_return - self.out))

            self.trainable_vars = tf.compat.v1.trainable_variables(self.var_scope)

            # contains gradients for all trainable variables
            self.gradient_placeholders = []
            for _ in range(len(self.trainable_vars)):
                self.gradient_placeholders.append(tf.compat.v1.placeholder(tf.float32))

            self.gradients = tf.gradients(self.loss, self.trainable_vars)

            self.optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.alpha, beta1=0.9, beta2=0.999
            )

            self.train_op = self.optimizer.apply_gradients(
                zip(self.gradient_placeholders, self.trainable_vars)
            )

    def init_session(self):
        sess = tf.compat.v1.Session()
        sess.run(tf.global_variables_initializer())
        gradient_adjustments = sess.run(
            tf.compat.v1.trainable_variables(self.var_scope)
        )
        for i, grad in enumerate(gradient_adjustments):
            gradient_adjustments[i] = grad * 0
        return sess, gradient_adjustments

    def __call__(self, s) -> float:
        s = np.reshape(s, (1, self.state_dims))
        return self.sess.run(self.out, feed_dict={self.state: s})

    def set_gradient_adjustments(self, s, G):
        gradient_adjustments = self.get_grads(s, G)
        for i, grad in enumerate(gradient_adjustments):
            self.gradient_adjustments[i] += grad

    def clear_gradient_adjustments(self):
        for i, grad in enumerate(self.gradient_adjustments):
            self.gradient_adjustments[i] += grad * 0

    def update(self, s, G):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        s = np.reshape(s, (1, self.state_dims))
        G = np.reshape(G, (1,))

        self.set_gradient_adjustments(s, G)

        # set feed for adjusting gradient for each training var
        feed = dict(zip(self.gradient_placeholders, self.gradient_adjustments))

        self.sess.run(self.train_op, feed_dict=feed)

        self.clear_gradient_adjustments()

        self.steps += 1

    def get_vars(self):
        return self.sess.run(tf.compat.v1.trainable_variables(self.var_scope))

    def get_grads(self, s, G):
        G = np.reshape(G, (1, 1))
        feed = {self.state: s, self.target_return: G}
        return self.sess.run(self.gradients, feed_dict=feed)


def REINFORCE(
    env,  # open-ai environment
    gamma: float,
    num_episodes: int,
    pi: PiApproximationWithNN,
    V: Baseline,
) -> Iterable[float]:
    """
    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    max_return = float("-inf")
    returns = []

    for episode in range(1, num_episodes + 1):
        if episode % 100 == 0:
            print(f"episode: {episode}")

        states, actions, Gs = [env.reset()], [], []
        s, t, done = states[0], 0, False

        while not done:
            a = pi(s)
            s, r, done, _ = env.step(a)

            states.append(s)
            actions.append(a)
            Gs.append(r)
            for i in range(1, len(Gs)):
                Gs[len(Gs) - i - 1] += (gamma ** i) * r
            t += 1

        T = t
        for t in range(T):
            # cumulative sum of rewards makes finding G easier
            # eg. total return from s_0 == acc_rewards[-1]
            # subtracting the accumulated rewards up until t
            # gives return at time t
            delta = Gs[t] - V(states[t])
            V.update(states[t], Gs[t])
            pi.update(states[t], actions[t], gamma ** t, delta)

        max_return = max(max_return, Gs[0])
        returns.append(Gs[0])
    print(f"max return: {max_return}")

    return returns
