#!/user/bin/env python

'''tic-tac-toe.py: Implement the game of Tic-Tac-Toe.'''

################################################################################

from ...game_if.game import Game

import numpy as np

class TicTacToe(Game):

    CIRCLE = 0
    CROSS = 1
    lines = [[(0, 0), (0, 1), (0, 2)], [(0, 0), (1, 0), (2, 0)], [(1, 0), (1, 1), (1, 2)], [(0, 1), (1, 1), (2, 1)], [(2, 0), (2, 1), (2, 2)], [(0, 2), (1, 2), (2, 2)], [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]]

    def __init__(self):
        self.nbr_players = 2
        self.board = np.zeros(shape=(3,3,3))
        self.board[2,:,:] = 1.
        self.player_turn = CIRCLE

    def get_action_list():
        return range(9)

    def take_action(self, player_id, action_id):
        if self.player_turn != player_id:
            raise Exception('It was not player_id %d\'s turn!'%player_id)

        old_status = self.eval_win_cond()

        if old_status >= 0:
            raise Exception('Game was already won by player %d!'%old_status)
        if old_status == -2:
            raise Exception('Game was already finished with a draw!')
        if old_status == -3:
            raise Exception('Game was already finished by an invalid move!')

        x,y = action_id_2_xy(action_id)

        self.board[2,x,y] -= 1
        self.board[player_id,x,y] += 1
        self.player_turn = 1-self.player_turn

        return self.eval_win_cond()

    def action_id_2_xy(id):
        return (id//3, id%3)

    def eval_win_cond(self):
        assert(np.max(self.board[2,:,:])<=1)
        assert(np.min(self.board[:2,:,:])>=0)
        assert(np.sum(self.board, axis=0)==np.ones([3,3]))

        if np.min(self.board[2,:,:])<0: # invalid move
            return -3

        circle_lines = 0
        cross_lines = 0

        for line in lines:
            circ_f = 1
            cros_f = 1
            for cell in line:
                circ_f *= self.board[CIRCLE][cell]
                cros_f *= self.board[CROSS][cell]
            circle_lines += circ_f
            cross_lines += cros_f

        assert(circle_lines <= 1)
        assert(cross_lines <= 1)
        assert(cross_lines * circle_lines == 0)

        if cross_lines == 0 and circle_lines == 0:
            if np.max(self.board[2,:,:]) == 0:  # no possible move left, it is a draw
                return -2
            else:   # game is still ongoing
                return -1

        return CIRCLE*circle_lines + CROSS*cross_lines
