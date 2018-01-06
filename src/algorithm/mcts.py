#!/user/bin/env python
'''mcts.py: Implement the Monte-Carlo tree search based on a network policy.'''
###############################################################################

import logging
import logging.config
import numpy as np
from copy import deepcopy as copy

logging.config.fileConfig('../cfg/logging.cfg')
logger = logging.getLogger('mcts_visualize')


class Tree_Node:
    def __init__(self, game, mcts, parent=None, parent_action=None):
        self.mcts = mcts
        self.parent = parent
        self.parent_action = parent_action
        self.childs = [None] * game.max_nbr_actions
        self.action_list = game.get_action_list()
        self.game = copy(game)
        if not self.game.is_terminal():
            self.terminal = False
            probs, values = mcts.mcts_net.evaluate(game.get_state_for_player(
                game.get_player_turn()))
            self.prior_probability = probs
            self.values = values
            self.visit_count = [0] * game.max_nbr_actions
            self.total_action_value = [0] * game.max_nbr_actions
            self.mean_action_value = [1. / game.nbr_players] \
                * game.max_nbr_actions
        else:
            self.terminal = True
            self.values = self.game.get_points()
            for i in range(self.game.get_player_turn()):
                self.values = list(self.values[1:]) + [self.values[0]]

    def select(self):
        total_visit_count = sum(self.visit_count)
        if total_visit_count == 0:
            QpU = self.prior_probability
        else:
            U = self.mcts.c_puct * np.sqrt(total_visit_count) * np.array(
                self.prior_probability) / (1. + np.array(self.visit_count))
            QpU = np.array(self.mean_action_value) + U
        return self.action_list[np.argmax(QpU[self.action_list])]

    def expand_eval(self, action_id):
        new_game = copy(self.game)
        new_game.take_action(new_game.get_player_turn(), action_id)
        new_node = Tree_Node(new_game, self.mcts,
                             parent=self, parent_action=action_id)
        self.childs[action_id] = new_node

    def backup(self, values=None, action_id=None):
        if values is not None:
            assert action_id is not None
            self.visit_count[action_id] += 1
            self.total_action_value[action_id] += values[0]
            self.mean_action_value[action_id] = \
                self.total_action_value[action_id] / \
                self.visit_count[action_id]
        else:
            values = self.values
        if self.parent is not None:
            assert self.parent_action is not None
            values = [values[-1]] + list(values[:-1])
            self.parent.backup(values=values, action_id=self.parent_action)

    def get_probabilities(self):
        vcount = np.array(self.visit_count, dtype=np.float32)
        if len(self.action_list) > 1:
            if np.max(vcount) == np.sum(vcount):
                print('ALARM!################################################')
        print('%d, relative:%f' % (np.max(vcount),
                                   np.max(vcount) / np.sum(vcount)))
        if self.mcts.temperature == 0:
            probs = np.array(vcount == np.max(vcount), dtype=np.float32)
        else:
            invtemp = 1. / self.mcts.temperature
            vcount /= np.max(vcount)
            probs = vcount ** invtemp
        return probs / np.sum(probs)

    def leaves(self):
        if self.terminal:
            return 1
        if sum(self.visit_count) == 0:
            return 1
        result = 0
        for c in self.childs:
            if c is not None:
                result += c.leaves()
        return result

    def visualize(self):
        """Returns the subtree in ASCII Art

        Each entry corresponds to one line of the node's sub tree
        in ASCII Art.
        """
        if self.parent is None:
            P = '  '
            N = '  '
            Q = '  '
        else:
            P = '%02d' % round(self.parent.prior_probability[self.parent_action]*100)
            N = '%02d' % self.parent.visit_count[self.parent_action]
            Q = '%02d' % round(self.parent.mean_action_value[self.parent_action]*100)
        V = '%02d' % round(self.values[0] * 100)
        my_contribution = ['%s,%s,%s,%s' % (P[-2:],V[-2:],N[-2:],Q[-2:])] + \
            self.game.minimal_visualize_arr()
        child_contributions = ['']
        for child in self.childs:
            if child is None:
                continue
            cur_contribution = child.visualize()
            while len(child_contributions) < len(cur_contribution):
                child_contributions.append(' ' * len(child_contributions[0]))
            while len(child_contributions) > len(cur_contribution):
                cur_contribution.append(' ' * len(cur_contribution[0]))
            for i in range(len(child_contributions)):
                child_contributions[i] += ' ' + cur_contribution[i]
        if len(child_contributions[0]) == 0:
            return my_contribution
        for i in range(len(child_contributions)):
            child_contributions[i] = child_contributions[i][1:]

        my_contribution.append(' ' * len(my_contribution[0]))
        while len(child_contributions[0]) > len(my_contribution[0]):
            for i in range(len(my_contribution)):
                my_contribution[i] += ' '
        return my_contribution + child_contributions


class MCTS:
    def __init__(self, mcts_net, player_id, nbr_sims=32, temperature=1,):
        self.mcts_net = mcts_net
        self.player_id = player_id
        self.nbr_sims = nbr_sims
        self.temperature = temperature
        self.root = None
        self.c_puct = 20.0

    def evaluate(self, game):
        if self.root is None:
            self.root = Tree_Node(game, self)
        print('initial visitcount:%d' % sum(self.root.visit_count))
        for i in range(self.nbr_sims):
            node = self.root
            while True:
                a = node.select()
                if node.childs[a] is not None:
                    if node.childs[a].terminal:
                        break
                    node = node.childs[a]
                else:
                    break
            if node.childs[a] is None:
                node.expand_eval(a)
            node.childs[a].backup()
        self.visualize()
        return self.root.get_probabilities()

    def cut_root(self, game, action):
        if self.root is None:
            return
        print('action_count:%d' % self.root.visit_count[action])
        assert (game.get_state_for_player(0)
                == self.root.game.get_state_for_player(0)).all()
        self.root = self.root.childs[action]
        if self.root is not None:
            try:
                print('after cut: %d' % sum(self.root.visit_count))
            except Exception:
                print('after cut: terminal')
            self.root.parent = None
            self.root.parent_action = None

    def visualize(self):
        string_list = self.root.visualize()
        logger.debug('-----Visualizing the MCTS-----')
        for line in string_list:
            logger.debug(line)
