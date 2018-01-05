#!/user/bin/env python
'''
ultimate_tictactoe.py: Implement the game of Ultimate Tic-Tac-Toe.
'''
###############################################################################
import numpy as np


class Game():
    CIRCLE = 0
    CROSS = 1
    lines = [[(0, 0), (0, 1), (0, 2)], [(0, 0), (1, 0), (2, 0)],
             [(1, 0), (1, 1), (1, 2)], [(0, 1), (1, 1), (2, 1)],
             [(2, 0), (2, 1), (2, 2)], [(0, 2), (1, 2), (2, 2)],
             [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]]
    nbr_players = 2
    state_dim = 81 * 3
    max_nbr_actions = 81

    def __init__(self):
        self.board = np.zeros(shape=(3, 3, 3, 3, 3))
        self.superboard = np.zeros(shape=(3, 3, 3))
        self.board[2, :, :, :, :] = 1.
        self.superboard[2, :, :] = 1.
        self.player_turn = self.CIRCLE
        # if last_move == -1 then you can place anywhere
        self.last_move = -1

    def visualize(self):
        xorder = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        yorder = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        ords = ord(' ') * self.board[2, xorder, yorder, :, :]
        ords += ord('O') * self.board[0, xorder, yorder, :, :]
        ords += ord('X') * self.board[1, xorder, yorder, :, :]
        ords = np.array(ords, dtype=int)
        print('---------------------------------------')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' %
              tuple([chr(s) for s in ords[0:3, 0, :].flatten()]))
        print('---+---+---   ---+---+---   ---+---+---')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' %
              tuple([chr(s) for s in ords[0:3, 1, :].flatten()]))
        print('---+---+---   ---+---+---   ---+---+---')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' %
              tuple([chr(s) for s in ords[0:3, 2, :].flatten()]))
        print('                                       ')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' %
              tuple([chr(s) for s in ords[3:6, 0, :].flatten()]))
        print('---+---+---   ---+---+---   ---+---+---')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' %
              tuple([chr(s) for s in ords[3:6, 1, :].flatten()]))
        print('---+---+---   ---+---+---   ---+---+---')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' %
              tuple([chr(s) for s in ords[3:6, 2, :].flatten()]))
        print('                                       ')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' %
              tuple([chr(s) for s in ords[6:9, 0, :].flatten()]))
        print('---+---+---   ---+---+---   ---+---+---')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' %
              tuple([chr(s) for s in ords[6:9, 1, :].flatten()]))
        print('---+---+---   ---+---+---   ---+---+---')
        print(' %s | %s | %s     %s | %s | %s     %s | %s | %s ' %
              tuple([chr(s) for s in ords[6:9, 2, :].flatten()]))
        print('---------------------------------------')

    def minimal_visualize(self):
        xorder = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        yorder = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        ords = ord(' ') * self.board[2, xorder, yorder, :, :]
        ords += ord('O') * self.board[0, xorder, yorder, :, :]
        ords += ord('X') * self.board[1, xorder, yorder, :, :]
        ords = np.array(ords, dtype=int)
        print('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[0:3, 0, :].flatten()]))
        print('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[0:3, 1, :].flatten()]))
        print('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[0:3, 2, :].flatten()]))
        print('-----------')
        print('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[3:6, 0, :].flatten()]))
        print('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[3:6, 1, :].flatten()]))
        print('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[3:6, 2, :].flatten()]))
        print('-----------')
        print('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[6:9, 0, :].flatten()]))
        print('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[6:9, 1, :].flatten()]))
        print('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[6:9, 2, :].flatten()]))

    def minimal_visualize_arr(self):
        xorder = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        yorder = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        ords = ord(' ') * self.board[2, xorder, yorder, :, :]
        ords += ord('O') * self.board[0, xorder, yorder, :, :]
        ords += ord('X') * self.board[1, xorder, yorder, :, :]
        ords = np.array(ords, dtype=int)
        char_arr = []
        char_arr.append('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[0:3, 0, :].flatten()]))
        char_arr.append('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[0:3, 1, :].flatten()]))
        char_arr.append('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[0:3, 2, :].flatten()]))
        char_arr.append('-----------')
        char_arr.append('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[3:6, 0, :].flatten()]))
        char_arr.append('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[3:6, 1, :].flatten()]))
        char_arr.append('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[3:6, 2, :].flatten()]))
        char_arr.append('-----------')
        char_arr.append('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[6:9, 0, :].flatten()]))
        char_arr.append('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[6:9, 1, :].flatten()]))
        char_arr.append('%s%s%s|%s%s%s|%s%s%s' %
              tuple([chr(s) for s in ords[6:9, 2, :].flatten()]))
        return char_arr

    def get_player_turn(self):
        return self.player_turn

    def take_action(self, player_id, action_id):
        if self.player_turn != player_id:
            raise Exception('It was not player_id %d\'s turn!' % player_id)

        old_status = self.__get_status__()

        if old_status >= 0:
            raise Exception('Game was already won by player %d!' % old_status)
        if old_status == -2:
            raise Exception('Game was already finished with a draw!')
        if old_status == -3:
            raise Exception('Game was already finished by an invalid move!')

        xs, ys, x, y = self.__action_id_2_xy__(action_id)

        if action_id not in self.get_action_list():
            self.last_move = -2
        elif self.superboard[2, xs, ys] == 1.:
            self.board[2, xs, ys, x, y] -= 1
            self.board[player_id, xs, ys, x, y] += 1
            self.player_turn = 1 - self.player_turn
            self.last_move = action_id
            self.__evaluate_superboard__(xs, ys)
        else:
            self.last_move = -2

    def is_terminal(self):
        return self.__get_status__() != -1

    def get_points(self):
        assert self.is_terminal()
        status = self.__get_status__()
        if status == -2:
            result = [1.0 / self.nbr_players] * self.nbr_players
        elif status == -3:
            result = [1.0 / (self.nbr_players - 1)] * self.nbr_players
            result[self.get_player_turn()] = 0.0
        else:
            result = [0.0] * self.nbr_players
            result[status] = 1.0
        return result

    def __get_status__(self):
        assert(np.max(self.board[2, :, :, :, :]) <= 1)
        assert(np.min(self.board[:2, :, :, :, :]) >= 0)
        assert(np.all(np.sum(self.board, axis=0) == np.ones([3, 3, 3, 3])))

        if np.min(self.board[2, :, :, :, :]) < 0:  # invalid move
            return -3
        if self.last_move == -2:
            # also invalid move, because move to invalid global square
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
            # no possible move left, it is a draw
            if np.max(self.superboard[2, :, :]) == 0:
                return -2
            else:   # game is still ongoing
                return -1

        if cross_lines > 0:
            return self.CROSS
        elif circle_lines > 0:
            return self.CIRCLE
        else:
            raise Exception('Logic Error!')

    def __evaluate_superboard__(self, xs, ys):
        assert(np.max(self.board[2, xs, ys, :, :]) <= 1)
        assert(np.min(self.board[:2, xs, ys, :, :]) >= 0)
        assert(np.all(np.sum(self.board, axis=0) == np.ones([3, 3, 3, 3])))

        if np.min(self.board[2, xs, ys, :, :]) < 0:  # invalid move
            return
        if self.last_move == -2:
            # also invalid move, because move to invalid global square
            return

        circle_lines = 0
        cross_lines = 0

        for line in self.lines:
            circ_f = 1
            cros_f = 1
            for cell in line:
                circ_f *= self.board[self.CIRCLE][xs, ys][cell]
                cros_f *= self.board[self.CROSS][xs, ys][cell]
            circle_lines += circ_f
            cross_lines += cros_f

        assert(cross_lines * circle_lines == 0)

        if cross_lines == 0 and circle_lines == 0:
            # no possible move left, it is a draw
            if np.max(self.board[2, xs, ys, :, :]) == 0:
                self.superboard[2, xs, ys] = 0.
                return
            else:   # game is still ongoing
                return

        if cross_lines > 0:
            self.superboard[2, xs, ys] = 0.
            self.superboard[1, xs, ys] = 1.
            return
        elif circle_lines > 0:
            self.superboard[2, xs, ys] = 0.
            self.superboard[0, xs, ys] = 1.
            return
        else:
            raise Exception('Logic Error!')

    def get_action_list(self):
        poss_actions = []
        xss, yss = (-1, -1)
        if(self.last_move >= 0):
            lxs, lys, xss, yss = self.__action_id_2_xy__(self.last_move)
            if self.superboard[2, xss, yss] == 0.:
                xss, yss = (-1, -1)
        for x in range(3):
            for y in range(3):
                if xss == -1:
                    for xs in range(3):
                        for ys in range(3):
                            if self.board[2, xs, ys, x, y] == 1. \
                                    and self.superboard[2, xs, ys] == 1.:
                                poss_actions.append(
                                    self.__xy_2_action_id__(xs, ys, x, y))
                else:
                    if self.board[2, xss, yss, x, y] == 1. \
                            and self.superboard[2, xss, yss] == 1.:
                        poss_actions.append(
                            self.__xy_2_action_id__(xss, yss, x, y))
        return poss_actions

    def get_state_for_player(self, player_id):
        if player_id == 0:
            return self.board.flatten()
        elif player_id == 1:
            return self.board[[1, 0, 2]].flatten()
        else:
            raise Exception('The player %d does not exist!' % player_id)

    def __action_id_2_xy__(self, a_id):
        return ((a_id // 27) % 3, (a_id // 9) % 3,
                (a_id // 3) % 3, (a_id // 1) % 3)

    def __xy_2_action_id__(self, xs, ys, x, y):
        return xs * 27 + ys * 9 + x * 3 + y

    def user_input_2_action(self):
        print('xs, ys, x, y?')
        xs, ys, x, y = (int(num) for num in input().split())
        return self.__xy_2_action_id__(xs, ys, x, y)

    def action_2_user_output(self, action):
        xs, ys, x, y = self.__action_id_2_xy__(action)
        return '(%d, %d, %d, %d)' % (xs, ys, x, y)


if __name__ == '__main__':
    g = Game()
    turn = 0

    while not g.is_terminal():
        g.visualize()
        print(g.get_action_list())
        action = g.user_input_2_action()
        g.take_action(turn, action)
        turn = 1 - turn

    g.visualize()
    print(g.get_points())
