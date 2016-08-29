import numpy as np
import random as rand
import types


class SingleQLearner(object):

    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=None,  # learning rate
                 gamma=0.9,  # discount rate
                 rar=0.5,  # random action probability
                 radr=0.99,  # random action decay rate
                 dyna=0,  # number of dyna episodes
                 verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = lambda s, a: 1 / self.n[s, a] if not alpha else alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.n = np.zeros((num_states, num_actions))
        self.q_table = np.ones((num_states, num_actions))

        # dyna tables
        self.exps = []
        self.d_rewards = np.zeros((num_states, num_actions))


# update rule :
# Q(s, a) = Q(s, a) + alpha[r + gamma * argmax_a'(Q(s', a') - Q(s, a))]

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        if (rand.random() < self.rar):
            action = rand.randint(0, self.num_actions - 1)
            if self.verbose:
                print('random action selected')

        else:
            action = (self.q_table[s, :]).argmax()

        if self.verbose:
            print("s =", s, "a =", action)
        return action

    def update_Q(self, s, a, s_prime, r):
        """
        @summary:  the Q table and return an action
        @param s_prime: Integer. The new state
        @param r: Float. Real valued immediate reward
        @returns: The selected action
        """
        s_prime = int(s_prime)
        # get current Q(s, a) value
        q_sa = self.q_table[s, a]
        # save past state and action taken
        old_state = s
        old_a = a

        # update n visits table:
        self.n[old_state, old_a] += 1

        # get the alpha value
        if isinstance(self.alpha, types.FunctionType):
            q_alpha = self.alpha(old_state, old_a)
        else:
            q_alpha = self.alpha

        # dyna table updates
        if self.dyna:
            # update experience tuples
            self.exps.append((old_state, old_a, s_prime))

            # update R table
            r_sa = self.d_rewards[old_state, old_a]
            self.d_rewards[old_state, old_a] = \
                ((1 - q_alpha) * r_sa) + q_alpha * r

        # update Q(s, a)
        self.q_table[old_state, old_a] = ((1 - q_alpha) * q_sa) + \
            q_alpha * (r + self.gamma * self.q_table[s_prime, :].max())

        # update random action rate
        self.rar *= self.radr

        if (self.dyna > 0):
            self.dyna_iters()

        # get the next action
        action = self.querysetstate(s_prime)

        return self.q_table[s_prime, :]

    def dyna_iters(self):
        idxs = np.random.randint(len(self.exps), size=self.dyna)
        for i in idxs:
            self.make_dyna_updates(*self.exps[i])
        return

    def make_dyna_updates(self, dyna_s, dyna_a, dyna_s_prime):
        if isinstance(self.alpha, types.FunctionType):
            alpha = self.alpha[old_state, old_a]
        else:
            alpha = self.alpha

        # get reward for s, a
        r_sa = self.d_rewards[dyna_s, dyna_a]
        # get Q[s, a]
        dq_sa = self.q_table[dyna_s, dyna_a]
        # get dyna a'
        dyna_a_prime = self.q_table[dyna_s_prime, :].argmax()

        # dyna Q updates
        self.q_table[dyna_s, dyna_a] = \
            ((1 - alpha) * dq_sa) + alpha * (
            r_sa + self.gamma * self.q_table[dyna_s_prime, dyna_a_prime]
        )
        return

if __name__ == "__main__":
    print("420")
