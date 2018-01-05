#!/user/bin/env python

'''tic-tac-toe.py: Implement the game of Tic-Tac-Toe.'''

################################################################################

import numpy as np

class Game():

    CIRCLE = 0
    CROSS = 1
    lines = [[(0, 0), (0, 1), (0, 2)], [(0, 0), (1, 0), (2, 0)], [(1, 0), (1, 1), (1, 2)], [(0, 1), (1, 1), (2, 1)], [(2, 0), (2, 1), (2, 2)], [(0, 2), (1, 2), (2, 2)], [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]]
    nbr_players = 2
    state_dim = 27
    max_nbr_actions = 9

    def __init__(self):
        self.board = np.zeros(shape=(3,3,3))
        self.board[2,:,:] = 1.
        self.player_turn = self.CIRCLE

    def visualize(self):
        ords = ord(' ')*self.board[2,:,:]
        ords += ord('O')*self.board[0,:,:]
        ords += ord('X')*self.board[1,:,:]
        ords = np.array(ords, dtype=int)
        print(' %s | %s | %s '%tuple([chr(s) for s in ords[0]]))
        print('---+---+---')
        print(' %s | %s | %s '%tuple([chr(s) for s in ords[1]]))
        print('---+---+---')
        print(' %s | %s | %s '%tuple([chr(s) for s in ords[2]]))

    def get_player_turn(self):
        return self.player_turn

    def take_action(self, player_id, action_id):
        if self.player_turn != player_id:
            raise Exception('It was not player_id %d\'s turn!'%player_id)

        old_status = self.get_status()

        if old_status >= 0:
            raise Exception('Game was already won by player %d!'%old_status)
        if old_status == -2:
            raise Exception('Game was already finished with a draw!')
        if old_status == -3:
            raise Exception('Game was already finished by an invalid move!')

        x,y = self.action_id_2_xy(action_id)

        self.board[2,x,y] -= 1
        self.board[player_id,x,y] += 1
        self.player_turn = 1-self.player_turn

    def get_status(self):
        assert(np.max(self.board[2,:,:])<=1)
        assert(np.min(self.board[:2,:,:])>=0)
        assert(np.all(np.sum(self.board, axis=0)==np.ones([3,3])))

        if np.min(self.board[2,:,:])<0: # invalid move
            return -3

        circle_lines = 0
        cross_lines = 0

        for line in self.lines:
            circ_f = 1
            cros_f = 1
            for cell in line:
                circ_f *= self.board[self.CIRCLE][cell]
                cros_f *= self.board[self.CROSS][cell]
            circle_lines += circ_f
            cross_lines += cros_f

        assert(cross_lines * circle_lines == 0)

        if cross_lines == 0 and circle_lines == 0:
            if np.max(self.board[2,:,:]) == 0:  # no possible move left, it is a draw
                return -2
            else:   # game is still ongoing
                return -1

        if cross_lines > 0:
            return self.CROSS
        elif circle_lines > 0:
            return self.CIRCLE
        else:
            raise Exception('Logic Error!')

    def get_action_list(self):
        return range(9)

    def get_state_for_player(self, player_id):
        if player_id == 0:
            return self.board.flatten()
        elif player_id == 1:
            return self.board[[1,0,2]].flatten()
        else:
            raise Exception('The player %d does not exist!'%player_id)

    def action_id_2_xy(self, a_id):
        return (a_id//3, a_id%3)

if __name__ == '__main__':
    g = Game()
    turn = 0
    stat = g.get_status()

    while stat==-1:
        g.visualize()
        x,y = input().split(' ')
        act = 3*int(x)+int(y)
        g.take_action(turn, act)
        stat = g.get_status()
        turn = 1-turn

    g.visualize()
