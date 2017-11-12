#!/user/bin/env python
'''mcts.py: Implement the Monte-Carlo tree search based on a network policy.'''
###############################################################################

import numpy as np
from copy import deepcopy as copy


class Tree_Node:
    def __init__(self, game, mcts, parent=None, parent_action=None):
        self.mcts = mcts
        self.parent = parent
        self.parent_action = parent_action
        self.childs = [None] * game.max_nbr_actions
        self.action_list = game.get_action_list()
        self.game = copy(game)
        probs, values = mcts.network_func(game)
        self.prior_probability = probs
        self.visit_count = [0] * game.max_nbr_actions
        self.total_action_value = [0] * game.max_nbr_actions
        self.mean_action_value = [1. / game.nbr_players] * game.max_nbr_actions
        self.values = values

    def select(self):
        total_visit_count = sum(self.visit_count.values())
        if total_visit_count == 0:
            QpU = self.prior_probability
        else:
            U = self.mcts.c_puct * np.sqrt(total_visit_count) * np.array(
                self.prior_probability) / (1. + np.array(self.visit_count))
            QpU = np.array(self.mean_action_value) + U
        return self.action_list[np.argmax(QpU[self.action_list])]

    def expand_eval(self, action_id):
        new_game = copy(self.game)
        new_game.take_action(action_id)
        new_node = Tree_Node(new_game, self.mcts,
                             parent=self, parent_action=action_id)
        self.childs[action_id] = new_node

    def backup(self, values=None, value_id=None, action_id=None):
        if values:
            assert(action_id)
            assert(value_id)
            self.visit_count[action_id] += 1
            self.total_action_value[action_id] += values[value_id]
            self.mean_action_value[action_id] = \
                self.total_action_value[action_id] / \
                self.visit_count[action_id]
        else:
            values = self.values
            value_id = 1
        if self.parent:
            assert(self.parent_action)
            value_id = (self.game.nbr_players + value_id -
                        1) % self.game.nbr_players
            self.parent.backup(values=values, value_id=value_id,
                               action_id=self.parent_action)

    def get_probabilities(self):
        invtemp = 1. / self.mcts.temperature
        vcount = np.array(self.visit_count)
        if self.temp == 0:
            probs = np.array(vcount == np.max(vcount), dtype=np.float32)
        else:
            vcount /= np.max(vcount)
            probs = vcount ** invtemp
        return probs / np.sum(probs)


class MCTS:
    def __init__(self, network_func, player_id, logger_list=None, nbr_sims=30,
                 temperature=1,):
        self.network_func = network_func
        self.player_id = player_id
        self.logger_list = logger_list
        self.nbr_sims = nbr_sims
        self.temperature = temperature
        self.root = None

    def ai(self, game, opp_action_list):
        assert(len(opp_action_list) + 1 == game.nbr_players)
        for a in opp_action_list:
            if self.root:
                self.root = self.root.childs[a]
                self.root.parent = None
                self.root.parent_action = None
        if self.root is None:
            self.root = Tree_Node(game, self)
        for i in range(self.nbr_sims):
            node = self.root
            while True:
                a = node.select()
                if node.childs[a] is not None:
                    node = node.childs[a]
                else:
                    break
            node.expand_eval(a)
            node.childs[a].backup()
        probs = self.root.get_probabilities()
        if self.logger_list:
            self.logger_list.append(
                (game.get_state_for_player(self.player_id), probs))
        selected_action = np.random.choice(game.max_nbr_actions, p=probs)
        self.root = self.root.childs[selected_action]
        self.root.parent = None
        self.root.parent_action = None
        return selected_action
