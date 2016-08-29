import numpy as np
import itertools

np.random.seed(420)

N_ACTIONS = 5
N_STATES = 8
GOAL_A = [0, 4]
GOAL_B = [3, 7]

class SoccerGame(object):

    def __init__(self, n_states=N_STATES):
        self.A = 2
        self.B = 1
        # assign the ball randomly
        self.ball = self.get_mv_order()[0]
        # haha fuck that
        self.ball = 1
        all_states = list(itertools.product(range(n_states), range(n_states), range(2)))
        all_states = list(filter(lambda x: x[0] != x[1], all_states))
        self.state_lu = {k: v for v, k in enumerate([str(s0) + str(s1) + str(s2) for s0, s1, s2 in all_states])}

    def __repr__(self):
        return 'A Position: {}, B Position: {}, Ball: {}'.format(self.A, self.B, self.ball)

    @staticmethod
    def get_mv_order():
        if np.random.randint(2) == 0:
            return ['A', 'B']
        return ['B', 'A']

    def get_rewards(self):
        if ((self.A in GOAL_A) and (self.ball == 1)) or (
        (self.B in GOAL_A) and (self.ball == 0)):
            return (100, -100)
        if ((self.B in GOAL_B) and (self.ball == 0)) or (
        (self.A in GOAL_B) and (self.ball == 1)):
            return (-100, 100)
        return (0, 0)

    def chk_collision(self, a_pos=None, b_pos=None):
        if a_pos is None:
            a_pos = self.A
        if b_pos is None:
            b_pos = self.B
        return a_pos == b_pos

    def resolve_collision(self, mv_order, action_a, action_b):
        # this probably isn't correct right now
        if action_a == 4:
            self.ball = 1
        elif action_b == 4:
            self.ball = 0
        else:
            self.ball = 1 if mv_order[-1] == 'A' else 0
        return

    def get_next_state(self, player, action):
        """
        gets the next state for a single player and a given action
        :param player: Player to perform action
        :type player: string, 'A' or 'B'

        :param action: Action for player to perform
        :type action: int, in (0, 4)

        :returns s_prime: the next state for the player
        :type s_prime: int
        """
        # actions {N, E, S, W, STAY} -> {0, 1, 2, 3, 4}
        s = int(self.__dict__[player])
        if action == 0:
            s_prime = s - 4 if s > 3 else s
        elif action == 1:
            s_prime = s + 1 if s not in [3, 7] else s
        elif action == 2:
            s_prime = s + 4 if s <= 3 else s
        elif action == 3:
            s_prime = s - 1 if s not in [0, 4] else s
        elif action == 4:
            s_prime = s
        return s_prime

    def parse_state(self, state_tup):
        s0, s1, s2 = state_tup
        state = str(s0) + str(s1) + str(s2)
        return self.state_lu[state]

    def get_next_states(self, actions, mv_order):
        """
        gets the next states for A and B assuming no collision.
        If collision occurs, then transfer ball possession appropriately
        :param actions: player: action pairs
        :type actions: dictionary
        """

        # actions {N, E, S, W, STAY} -> {0, 1, 2, 3, 4}
        next_states = {'A': self.A, 'B': self.B}

        for player, action in actions.items():
            next_states[player] = self.get_next_state(player, action)

            if self.chk_collision(next_states['A'], next_states['B']):
                self.resolve_collision(mv_order, actions['A'], actions['B'])
                break # no moves take place idt?
        else:
            self.A = next_states['A']
            self.B = next_states['B']

        return self.parse_state([self.A, self.B, self.ball])


    def run_episode(self, action_a, action_b):
        mv_order = self.get_mv_order()
        next_states = self.get_next_states({'A': action_a, 'B': action_b}, mv_order)
        return next_states, self.get_rewards()




    def build_t_mat(self):
        pos_acts = [{'A': a1, 'B': b1} for a1, b1 in itertools.product(range(5), repeat=2)]

        all_states = self.state_lu.values()

        t_mat = np.zeros((112, 5, 5, 112))

        def sim_run(acts):
            """returns list of state, """
            all_states = self.state_lu.values()
            unparsed = list(map(lambda x: ' '.join([y for y in x]).split() , self.state_lu.keys()))
            s_primes = []
            for unps, pst in zip(unparsed, all_states):
                self.A, self.B, self.ball = unps

                s_primes.append((pst, acts, self.run_episode(acts['A'], acts['B'])[0]))
            return s_primes

        for _ in range(1000):
            for act in pos_acts:
                ress = sim_run(act)
                for res in ress:
                    t_mat[res[0], res[1]['A'], res[1]['B'], res[-1]] += 1

        t_mat /= t_mat.sum(axis=(-1), keepdims=True)
        return t_mat

