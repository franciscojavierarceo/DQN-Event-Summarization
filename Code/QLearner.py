import numpy as np
import random as rand


class QLearner(object):

    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,  # learning rate
                 gamma=0.9,  # discount rate
                 rar=0.5,  # random action probability
                 radr=0.99,  # random action decay rate
                 dyna=0,  # number of dyna episodes
                 verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        # self.q_table = np.random.randn(num_states, num_actions) / 10
        # self.q_table = np.zeros((num_states, num_actions))
        self.q_table = np.ones((num_states, num_actions))
        # self.exps = np.array(dtype=(int, int, int))

        # dyna tables
        self.exps = []
        self.d_rewards = np.zeros((num_states, num_actions))

        self.s = 0
        self.a = 0

# update rule :
# Q(s, a) = Q(s, a) + alpha[r + gamma * argmax_a'(Q(s', a') - Q(s, a))]

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        self.s = s

        if (rand.random() < self.rar):
            action = rand.randint(0, self.num_actions - 1)
            if self.verbose:
                print 'random action selected'

        else:
            action = np.argmax(self.q_table[s, :])

        if self.verbose:
            print "s =", s, "a =", action
        self.a = action
        return action

    def query(self, s_prime, r):
        """
        @summary:  the Q table and return an action
        @param s_prime: Integer. The new state
        @param r: Float. Real valued immediate reward
        @returns: The selected action
        """
        # get current Q(s, a) value
        q_sa = self.q_table[self.s, self.a]
        # save past state and action taken
        old_state = self.s
        old_a = self.a

        # dyna table updates
        if self.dyna:
            # update experience tuples
            self.exps.append((old_state, old_a, s_prime))

            # update R table
            r_sa = self.d_rewards[old_state, old_a]
            self.d_rewards[old_state, old_a] = \
                ((1 - self.alpha) * r_sa) + self.alpha * r

        # update Q(s, a)
        self.q_table[old_state, old_a] = ((1 - self.alpha) * q_sa) + \
            self.alpha * (r + self.gamma * self.q_table[s_prime, :].max())

        # update random action rate
        self.rar *= self.radr

        if (self.dyna > 0):
            self.dyna_iters()

        # get the next action
        action = self.querysetstate(s_prime)

        if self.verbose:
            self.verbose_prints(r, old_state)
        return action

    def dyna_iters(self):
        idxs = np.random.randint(len(self.exps), size=self.dyna)
        for i in idxs:
            self.make_dyna_updates(*self.exps[i])
        return

    def make_dyna_updates(self, dyna_s, dyna_a, dyna_s_prime):
        # get reward for s, a
        r_sa = self.d_rewards[dyna_s, dyna_a]
        # get Q[s, a]
        dq_sa = self.q_table[dyna_s, dyna_a]
        # get dyna a'
        dyna_a_prime = self.q_table[dyna_s_prime, :].argmax()

        # dyna Q updates
        self.q_table[dyna_s, dyna_a] = \
            ((1 - self.alpha) * dq_sa) + self.alpha * (
            r_sa + self.gamma * self.q_table[dyna_s_prime, dyna_a_prime]
        )
        return

    def verbose_prints(self, r, old_state):
        print "s =", self.s, "a =", self.a, "r =", r
        print 'q_table[{}, :] : {}'.format(
            max([old_state - 1, 0]),
            self.q_table[max([old_state - 1, 0]), :]
        )
        print 'q_table[{}, :] : {}'.format(
            old_state,
            self.q_table[old_state, :]
        )
        print 'q_table[{}, :] : {}'.format(
            min([old_state + 1, 99]),
            self.q_table[min([old_state + 1, 99]), :]
        )
        print 'q_table[{}, :] : {}'.format(
            max([self.s - 1, 0]),
            self.q_table[max([self.s - 1, 0]), :]
        )
        print 'q_table[{}, :] : {}'.format(
            self.s,
            self.q_table[self.s, :]
        )
        print 'q_table[{}, :] : {}'.format(
            min([self.s + 1, 99]),
            self.q_table[min([self.s + 1, 99]), :]
        )

