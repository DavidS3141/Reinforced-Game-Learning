#!/usr/bin/env python
"""mcts.py: Implement the Monte-Carlo tree search based on a network policy."""
###############################################################################

import logging
import logging.config
import numpy as np
import os.path as osp
import os

logging.config.fileConfig(osp.join(os.getenv("CFG_DIR"), "logging.cfg"))
logger = logging.getLogger("mcts_visualize")


class MCTS:
    class Node:
        def __init__(self, state, mcts, parent=None, parent_action=None):
            self.mcts = mcts
            self.parent = parent
            self.parent_action = parent_action
            self.childs = [None] * self.mcts.game.max_nbr_actions
            self.action_list = self.mcts.game.get_action_list(state)
            self.state = state
            if not self.mcts.game.is_terminal(self.state):
                self.terminal = False
                probs, values = self.mcts.policy_value_function(
                    self.mcts.game.flatten_state(self.state)
                )
                self.prior_probability = probs
                self.values = values
                self.visit_count = [0] * self.mcts.game.max_nbr_actions
                self.total_action_value = [0] * self.mcts.game.max_nbr_actions
                self.mean_action_value = [
                    1.0 / self.mcts.game.nbr_players
                ] * self.mcts.game.max_nbr_actions
            else:
                self.terminal = True
                self.values = self.mcts.game.get_points(self.state)

        def select_next_action_id(self):
            total_visit_count = sum(self.visit_count)
            if total_visit_count == 0:
                QpU = np.array(self.prior_probability)
            else:
                U = (
                    self.mcts.c_puct
                    * np.sqrt(total_visit_count)
                    * np.array(self.prior_probability)
                    / (1.0 + np.array(self.visit_count))
                )
                QpU = np.array(self.mean_action_value) + U
            available_QpU = QpU[self.action_list]
            maxQpU = available_QpU.max()
            good_idxs = np.where(maxQpU == available_QpU)[0]
            return self.action_list[np.random.choice(good_idxs)]

        def expand_eval(self, action_id):
            new_state = self.mcts.game.take_action(
                self.state, self.mcts.game.get_player_turn(self.state), action_id
            )
            new_node = MCTS.Node(
                new_state, self.mcts, parent=self, parent_action=action_id
            )
            self.childs[action_id] = new_node

        def backup(self, values=None, action_id=None):
            if values is not None:
                assert action_id is not None
                self.visit_count[action_id] += 1
                self.total_action_value[action_id] += values[
                    self.mcts.game.get_player_turn(self.state)
                ]
                self.mean_action_value[action_id] = (
                    self.total_action_value[action_id] / self.visit_count[action_id]
                )
            else:
                assert action_id is None
                values = self.values
            if self.parent is not None:
                assert self.parent_action is not None
                self.parent.backup(values=values, action_id=self.parent_action)

        def get_probabilities(self):
            vcount = np.array(self.visit_count, dtype=np.float32)
            if self.mcts.temperature == 0:
                probs = np.array(vcount == np.max(vcount), dtype=np.float32)
            else:
                invtemp = 1.0 / self.mcts.temperature
                vcount /= np.max(vcount)
                probs = vcount ** invtemp
            probs = probs / probs.sum()
            mask = np.zeros_like(probs)
            mask[self.action_list] = 1.0
            subsampled_probs = mask * probs
            assert abs(subsampled_probs.sum() - 1.0) < 1e-3 or np.max(vcount) == 0
            return subsampled_probs / subsampled_probs.sum()

        def visualize(self):
            """Returns the subtree in ASCII Art

            Each entry corresponds to one line of the node's sub tree
            in ASCII Art.
            """
            if self.parent is None:
                P = "   "
                N = "   "
                Q = "   "
            else:
                P = "%03d" % round(
                    self.parent.prior_probability[self.parent_action] * 100
                )
                N = "%05d" % self.parent.visit_count[self.parent_action]
                Q = "%03d" % round(
                    self.parent.mean_action_value[self.parent_action] * 100
                )
            V = "%03d" % round(self.values[0] * 100)
            my_contribution = ["%s,%s,%s,%s" % (P, V, N, Q)]
            # assert len(my_contribution[0]) == 15
            if hasattr(self.mcts.game, "minimal_visualize"):
                game_viz = self.mcts.game.minimal_visualize(self.state).split("\n")
            else:
                game_viz = self.mcts.game.visualize(self.state).split("\n")
            while len(my_contribution[0]) > len(game_viz[0]):
                for i in range(len(game_viz)):
                    game_viz[i] += " "
            my_contribution += game_viz
            child_contributions = [""]
            for child in self.childs:
                if child is None:
                    continue
                cur_contribution = child.visualize()
                while len(child_contributions) < len(cur_contribution):
                    child_contributions.append(" " * len(child_contributions[0]))
                while len(child_contributions) > len(cur_contribution):
                    cur_contribution.append(" " * len(cur_contribution[0]))
                for i in range(len(child_contributions)):
                    child_contributions[i] += " " + cur_contribution[i]
            if len(child_contributions[0]) == 0:
                return my_contribution
            for i in range(len(child_contributions)):
                child_contributions[i] = child_contributions[i][1:]

            my_contribution.append(" " * len(my_contribution[0]))
            while len(child_contributions[0]) > len(my_contribution[0]):
                for i in range(len(my_contribution)):
                    my_contribution[i] += " "
            return my_contribution + child_contributions

    def __init__(
        self, game, policy_value_function, nbr_sims=32, temperature=1, c_puct=20.0
    ):
        self.game = game
        self.policy_value_function = policy_value_function
        self.nbr_sims = nbr_sims
        self.temperature = temperature
        self.root = None
        self.c_puct = c_puct

    def evaluate(self, state):
        if self.root is None:
            self.root = MCTS.Node(state, self)
        for _ in range(self.nbr_sims):
            node = self.root
            while True:
                action_id = node.select_next_action_id()
                if node.childs[action_id] is not None:
                    if node.childs[action_id].terminal:
                        break
                    node = node.childs[action_id]
                else:
                    break
            if node.childs[action_id] is None:
                node.expand_eval(action_id)
            node.childs[action_id].backup()
        # self.visualize()
        return self.root.get_probabilities(), self.root.mean_action_value

    def cut_root(self, action_id):
        if self.root is None:
            return
        self.root = self.root.childs[action_id]
        if self.root is not None:
            self.root.parent = None
            self.root.parent_action = None

    def reset(self):
        self.root = None

    def visualize(self):
        string_list = self.root.visualize()
        logger.debug("-----Visualizing the MCTS-----")
        for line in string_list:
            logger.debug(line)
