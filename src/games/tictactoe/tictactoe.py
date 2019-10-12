#!/usr/bin/env python3

import numpy as np
from games.game import Game, GameEngine
from players.basic_players import Random, Human


class TicTacToe(Game):

    CIRCLE = 0
    CROSS = 1
    lines = [
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1)],
        [(2, 0), (2, 1), (2, 2)],
        [(0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)],
    ]
    nbr_players = 2
    state_dimensionality = 27
    max_nbr_actions = 9

    def generate_initial_state(self):
        board = np.zeros(shape=(3, 3, 3))
        board[2, :, :] = 1
        return dict(board=board, turn=self.CIRCLE)

    def flatten_state(self, state):
        if state["turn"] == 0:
            return state["board"].flatten()
        elif state["turn"] == 1:
            return state["board"][[1, 0, 2]].flatten()
        else:
            raise Exception("The player %d does not exist!" % state["turn"])

    def visualize(self, state):
        ords = ord(" ") * state["board"][2, :, :]
        ords += ord("O") * state["board"][0, :, :]
        ords += ord("X") * state["board"][1, :, :]
        ords = np.array(ords, dtype=int)
        ords[
            np.logical_and(
                np.logical_and(ords != ord(" "), ords != ord("X")), ords != ord("O")
            )
        ] = ord("#")
        lines = []
        lines.append(" %s | %s | %s " % tuple([chr(s) for s in ords[0]]))
        lines.append("---+---+---")
        lines.append(" %s | %s | %s " % tuple([chr(s) for s in ords[1]]))
        lines.append("---+---+---")
        lines.append(" %s | %s | %s " % tuple([chr(s) for s in ords[2]]))
        return "\n".join(lines)

    def get_player_turn(self, state):
        return state["turn"]

    def take_action(self, state, player_id, action_id):
        if self.get_player_turn(state) != player_id:
            raise Exception("It was not player_id %d's turn!" % player_id)

        old_status = self.get_status(state)

        if old_status >= 0:
            raise Exception("Game was already won by player %d!" % old_status)
        if old_status == -2:
            raise Exception("Game was already finished with a draw!")
        if old_status == -3:
            raise Exception("Game was already finished by an invalid move!")

        x, y = self.action_id_2_xy(action_id)

        board = state["board"].copy()
        board[2, x, y] -= 1
        board[player_id, x, y] += 1
        return dict(board=board, turn=1 - state["turn"])

    def is_terminal(self, state):
        return self.get_status(state) != -1

    def get_points(self, state):
        status = self.get_status(state)
        assert status != -1
        if status == -3:  # invalid move, set status to winner id
            status = state["turn"]
        if status == 0:
            return [1.0, -1.0]
        if status == 1:
            return [-1.0, 1.0]
        if status == -2:
            return [0.0, 0.0]
        raise Exception("LogicError")

    def get_status(self, state):
        assert np.max(state["board"][2, :, :]) <= 1
        assert np.min(state["board"][:2, :, :]) >= 0
        assert np.all(np.sum(state["board"], axis=0) == np.ones([3, 3]))

        if np.min(state["board"][2, :, :]) < 0:  # invalid move
            return -3

        circle_lines = 0
        cross_lines = 0

        for line in self.lines:
            circ_f = 1
            cros_f = 1
            for cell in line:
                circ_f *= state["board"][self.CIRCLE][cell]
                cros_f *= state["board"][self.CROSS][cell]
            circle_lines += circ_f
            cross_lines += cros_f

        assert cross_lines * circle_lines == 0

        if cross_lines == 0 and circle_lines == 0:
            if (
                np.max(state["board"][2, :, :]) == 0
            ):  # no possible move left, it is a draw
                return -2
            else:  # game is still ongoing
                return -1

        if cross_lines > 0:
            return self.CROSS
        elif circle_lines > 0:
            return self.CIRCLE
        else:
            raise Exception("Logic Error!")

    def get_action_list(self, state):
        xy_valids = np.where(state["board"][2] == 1)
        return [self.xy_2_action_id(x, y) for x, y in zip(*xy_valids)]

    def action_id_2_xy(self, action_id):
        return (action_id // 3, action_id % 3)

    def xy_2_action_id(self, x, y):
        return 3 * x + y

    def user_input_2_action(self):
        print("Please provide your next turn through x,y input coordinates!")
        x, y = map(lambda x: int(x) - 1, input().split())
        return self.xy_2_action_id(x, y)

    def action_2_user_output(self, action_id):
        return str(tuple([v + 1 for v in self.action_id_2_xy(action_id)]))


if __name__ == "__main__":
    game = TicTacToe()
    players = [Random(game), Human(game)]
    engine = GameEngine(game, players)
    engine.run()
