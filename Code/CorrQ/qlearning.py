import numpy as np
from .SingleQLearner import SingleQLearner

class MultiQLearner(object):
    def __init__(self, num_states=112, num_actions=5, selection='u',
                alpha=lambda s, a: 1 / self.n[s, a], decay=1,
                gamma=.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.decay = decay
        self.gamma = gamma
        self.QL_A = SingleQLearner(num_states, num_actions, None, gamma)
        self.QL_B = SingleQLearner(num_states, num_actions, None, gamma)
        self.selection = selection
        return

    def get_actions(self, s, actions=None, s_prime=None, rs=None):
        if s_prime:
            r_A = rs[0]
            r_B = rs[-1]

            act_A = actions['A']
            act_B = actions['B']

            q_A = self.QL_A.update_Q(s, act_A, s_prime, r_A)
            q_B = self.QL_B.update_Q(s, act_B, s_prime, r_B)

        else:
            s_prime = s

        # un-pack list of s_primes, rs

        #s_prime_A = str(s_primes[0]) + str(s_primes[1])
        #s_prime_A += '1' if s_primes[-1] == 'A' else '0'
        #s_prime_B = str(s_primes[1]) + str(s_primes[0])
        #s_prime_B += '0' if s_primes[-1] == 'A' else '1'

        #s_prime_B = s_prime_A  # try this with same S rep

        if self.selection == 'u':
            actions = self.get_actions_u(s_prime, s_prime)
        if self.selection == 'friend':
            actions = self.get_actions_friend(s_prime, s_prime)

        elif self.selection == 'q':
            actions = self.get_actions_q(s_prime, s_prime)

        elif self.selection == 'e':
            actions = self.get_actions_e(s_prime, s_prime)

        elif self.selection == 'r':
            actions = self.get_actions_r(s_prime, s_prime)

        elif self.selection == 'l':
            actions = self.get_actions_l(s_prime, s_prime)

        return actions

    def get_actions_u(self, s_prime_A, s_prime_B):
        # maximize the sum of the players rewards

        q_A = self.QL_A.q_table[s_prime_A, :]
        q_B = self.QL_B.q_table[s_prime_B, :]

        # oh god this is horrible
        best_sum_r = -999
        a_a = None
        a_b = None
        all_sum_rs = []

        for a, r_a in enumerate(q_A):
            for b, r_b in enumerate(q_B):
                sum_r = r_a + r_b
                all_sum_rs.append((a, b, sum_r))

        all_sum_rs = np.asarray(all_sum_rs)
        idx = np.random.choice(
            all_sum_rs.shape[0],
            p= (1000 + all_sum_rs[:, -1]) / (1000 + all_sum_rs[:, -1]).sum()
            )
        a_a, a_b = all_sum_rs[idx, :-1]
        return {'A': a_a, 'B': a_b}


    def get_actions_friend(self, s_prime_A, s_prime_B):
        # maximize the reward for the player w/ the highest R? I think
        # how does this relate to rCE-Q?

        q_A = self.QL_A.q_table[s_prime_A, :]
        q_B = self.QL_B.q_table[s_prime_B, :]

        # oh god this is horrible
        # best_sum_r = -999
        best_a = -np.inf
        best_b = -np.inf
        a_a = None
        a_b = None
        all_sum_rs = []

        for a, r_a in enumerate(q_A):
            for b, r_b in enumerate(q_B):
                all_sum_rs.append((a, b, max(r_a, r_b)))

        all_sum_rs = np.asarray(all_sum_rs)

        a_a, a_b, summed = max(all_sum_rs, key=lambda x: x[-1])
        return {'A': a_a, 'B': a_b}


    def get_actions_q(self, s_prime_A, s_prime_B):
        q_A = self.QL_A.q_table[s_prime_A, :]
        q_B = self.QL_B.q_table[s_prime_B, :]
        return {'A': q_A.argmax(), 'B': q_B.argmax()}

#class MultiQLearner(object):
#    def __init__(self, num_states=112, num_actions=5, selection='u',
#                alpha=lambda s, a: 1 / self.n[s, a], decay=1,
#                gamma=.9):
#        self.num_states = num_states
#        self.num_actions = num_actions
#        self.alpha = alpha
#        self.decay = decay
#        self.gamma = gamma
#        # Q = Q[S, a_A, a_B]
#        self.Q = np.zeros(num_states, num_actions, num_actions)
#        self.n = np.zeros_like(self.Q)
#        self.selection = selection
#        return
#
#    def update_Q(self, s, a, s_prime, r):
