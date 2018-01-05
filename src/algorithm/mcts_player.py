#!/user/bin/env python
'''
algorithm/mcts_player.py: Implement a player class using the AlphaGo Zero
    approach.
'''
###############################################################################
from algorithm.mcts import MCTS

import numpy as np


class MCTS_Player(object):
    def __init__(self, mcts_network, explore=True):
        self.net = mcts_network
        self.log_game_states = []
        self.log_priors = []
        self.temperature = 1.0
        if not explore:
            self.temperature = 0.0

    def set_player_id(self, player_id):
        self.player_id = player_id
        self.mcts = MCTS(mcts_net=self.net, player_id=self.player_id,
                         temperature=self.temperature)

    def get_action(self, game):
        assert hasattr(self, 'player_id')
        assert hasattr(self, 'mcts')
        assert game.get_player_turn() == self.player_id
        probs = self.mcts.evaluate(game)
        self.log_game_states.append(game.get_state_for_player(self.player_id))
        self.log_priors.append(probs)
        selected_action = np.random.choice(game.max_nbr_actions, p=probs)
        return selected_action

    def set_next_action(self, game, action):
        assert hasattr(self, 'mcts')
        self.mcts.cut_root(game, action)

    def set_result(self, result):
        for i in range(self.player_id):
            result = result[1:] + [result[0]]
        self.log_values = [result] * len(self.log_game_states)
        self.result_set = True

    def extract_train_data(self):
        assert hasattr(self, 'result_set')
        assert hasattr(self, 'log_values')
        assert self.result_set
        n = len(self.log_game_states)
        assert n == len(self.log_priors)
        assert n == len(self.log_values)
        return self.log_game_states, self.log_priors, self.log_values
