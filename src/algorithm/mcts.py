#!/user/bin/env python

'''mcts.py: Implement the Monte-Carlo tree search based on a network policy.'''

################################################################################

import numpy as np

class Tree_Node:

    def __init__(self, ):
        self.childs = dict()

class MCTS:

    def __init__(self, network_func, player_id, logger_list = None, nbr_sims = 30, temperature = 1, '''params'''):
        self.network_func = network_func
        self.player_id = player_id
        self.logger_list = logger_list
        self.nbr_sims = nbr_sims
        self.temperature = temperature
        self.root = None

    def ai(self, game, opp_action_list):
        assert(len(opp_action_list)+1 == game.nbr_players)
        for a in opp_action_list:
            if self.root:
                self.root = self.root.childs[a]
        if not self.root:
            self.root = Tree_Node(game)
        for i in range(self.nbr_sims):
            act_lifo = LIFO()
            node = self.root
            while True:
                a = node.select()
                act_lifo.push(a)
                if node.childs[a]:
                    node = node.childs[a]
                else:
                    break
            node.expand_eval(a)
            node.childs[a].backup()
        probs = self.root.get_probs(self.temperature)
        self.logger_list.append((game.get_state(), probs))
        selected_action = np.random.choice(list(self.root.childs), p=probs)
        self.root = self.root.childs[selected_action]
        return selected_action
