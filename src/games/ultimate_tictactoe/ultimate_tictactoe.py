#!/user/bin/env python

'''ultimate_tictactoe.py: Implement the game of Ultimate Tic-Tac-Toe.'''

################################################################################

import numpy as np

class Ultimate_TicTacToe():

    CIRCLE = 0
    CROSS = 1
    lines = [[(0, 0), (0, 1), (0, 2)], [(0, 0), (1, 0), (2, 0)], [(1, 0), (1, 1), (1, 2)], [(0, 1), (1, 1), (2, 1)], [(2, 0), (2, 1), (2, 2)], [(0, 2), (1, 2), (2, 2)], [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]]
    nbr_players = 2
    state_dim = 81*3

    def __init__(self):
        self.board = np.zeros(shape=(3,3,3,3,3))
        self.superboard = np.zeros(shape=(3,3,3))
        self.board[2,:,:,:,:] = 1.
        self.superboard[2,:,:] = 1.
        self.player_turn = self.CIRCLE
        self.last_move = -1

    def visualize(self):
        xorder = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        yorder = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        ords = ord(' ')*self.board[2,xorder,yorder,:,:]
        ords += ord('O')*self.board[0,xorder, yorder, :,:]
        ords += ord('X')*self.board[1,xorder, yorder, :,:]
        ords = np.array(ords, dtype=int)
        print('---------------------------------------')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' % tuple([chr(s) for s in ords[0:3,0,:].flatten()]))
        print('---+---+---   ---+---+---   ---+---+---')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' % tuple([chr(s) for s in ords[0:3,1,:].flatten()]))
        print('---+---+---   ---+---+---   ---+---+---')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' % tuple([chr(s) for s in ords[0:3,2,:].flatten()]))
        print('                                       ')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' % tuple([chr(s) for s in ords[3:6,0,:].flatten()]))
        print('---+---+---   ---+---+---   ---+---+---')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' % tuple([chr(s) for s in ords[3:6,1,:].flatten()]))
        print('---+---+---   ---+---+---   ---+---+---')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' % tuple([chr(s) for s in ords[3:6,2,:].flatten()]))
        print('                                       ')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' % tuple([chr(s) for s in ords[6:9,0,:].flatten()]))
        print('---+---+---   ---+---+---   ---+---+---')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' % tuple([chr(s) for s in ords[6:9,1,:].flatten()]))
        print('---+---+---   ---+---+---   ---+---+---')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' % tuple([chr(s) for s in ords[6:9,2,:].flatten()]))
        print('---------------------------------------')

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

        xs,ys,x,y = self.action_id_2_xy(action_id)

        if action_id not in self.get_action_list():
            self.last_move = -2
        elif self.superboard[2,xs,ys]==1.:
            self.board[2,xs,ys,x,y] -= 1
            self.board[player_id,xs,ys,x,y] += 1
            self.player_turn = 1-self.player_turn
            self.last_move = action_id
            self.evaluate_superboard(xs,ys)
        else:
            self.last_move = -2

    def get_status(self):
        assert(np.max(self.board[2,:,:,:,:])<=1)
        assert(np.min(self.board[:2,:,:,:,:])>=0)
        assert(np.all(np.sum(self.board, axis=0)==np.ones([3,3,3,3])))

        if np.min(self.board[2,:,:,:,:])<0: # invalid move
            return -3
        if self.last_move == -2:             # also invalid move, because move to invalid global square
            return -3

        circle_lines = 0
        cross_lines = 0

        for line in self.lines:
            circ_f = 1
            cros_f = 1
            for cell in line:
                circ_f *= self.superboard[self.CIRCLE][cell]
                cros_f *= self.superboard[self.CROSS][cell]
            circle_lines += circ_f
            cross_lines += cros_f

        assert(cross_lines * circle_lines == 0)

        if cross_lines == 0 and circle_lines == 0:
            if np.max(self.superboard[2,:,:]) == 0:  # no possible move left, it is a draw
                return -2
            else:   # game is still ongoing
                return -1

        if cross_lines > 0:
            return self.CROSS
        elif circle_lines > 0:
            return self.CIRCLE
        else:
            raise Exception('Logic Error!')

    def evaluate_superboard(self, xs, ys):
        assert(np.max(self.board[2,xs,ys,:,:])<=1)
        assert(np.min(self.board[:2,xs,ys,:,:])>=0)
        assert(np.all(np.sum(self.board, axis=0)==np.ones([3,3,3,3])))

        if np.min(self.board[2,xs,ys,:,:])<0: # invalid move
            return
        if self.last_move == -2:             # also invalid move, because move to invalid global square
            return

        circle_lines = 0
        cross_lines = 0

        for line in self.lines:
            circ_f = 1
            cros_f = 1
            for cell in line:
                circ_f *= self.board[self.CIRCLE][xs,ys][cell]
                cros_f *= self.board[self.CROSS][xs,ys][cell]
            circle_lines += circ_f
            cross_lines += cros_f

        assert(cross_lines * circle_lines == 0)

        if cross_lines == 0 and circle_lines == 0:
            if np.max(self.board[2,xs,ys,:,:]) == 0:  # no possible move left, it is a draw
                self.superboard[2,xs,ys] = 0.
                return
            else:   # game is still ongoing
                return

        if cross_lines > 0:
            self.superboard[2,xs,ys] = 0.
            self.superboard[1,xs,ys] = 1.
            return
        elif circle_lines > 0:
            self.superboard[2,xs,ys] = 0.
            self.superboard[0,xs,ys] = 1.
            return
        else:
            raise Exception('Logic Error!')

    def get_action_list(self):
        poss_actions = []
        xss, yss = (-1, -1)
        if(self.last_move>=0):
            lxs,lys,xss,yss = self.action_id_2_xy(self.last_move)
            if self.superboard[2,xss,yss]==0.:
                xss, yss = (-1, -1)
        for x in range(3):
            for y in range(3):
                if xss == -1:
                    for xs in range(3):
                        for ys in range(3):
                            if self.board[2,xs,ys,x,y]==1. and self.superboard[2,xs,ys]==1.:
                                poss_actions.append(self.xy_2_action_id(xs,ys,x,y))
                else:
                    if self.board[2,xss,yss,x,y]==1. and self.superboard[2,xss,yss]==1.:
                        poss_actions.append(self.xy_2_action_id(xss,yss,x,y))
        return poss_actions

    def get_state_for_player(self, player_id):
        if player_id == 0:
            return self.board.flatten()
        elif player_id == 1:
            return self.board[[1,0,2]].flatten()
        else:
            raise Exception('The player %d does not exist!'%player_id)

    def action_id_2_xy(self, a_id):
        return ((a_id//27)%3, (a_id//9)%3,(a_id//3)%3, (a_id//1)%3)

    def xy_2_action_id(self, xs, ys, x, y):
        return xs*27+ys*9+x*3+y

if __name__ == '__main__':
    g = Ultimate_TicTacToe()
    turn = 0
    stat = g.get_status()

    while stat==-1:
        g.visualize()
        print(g.get_action_list())
        xs,ys,x,y = input().split(' ')
        act = 27*int(xs)+9*int(ys)+3*int(x)+int(y)
        g.take_action(turn, act)
        stat = g.get_status()
        turn = 1-turn

    g.visualize()
