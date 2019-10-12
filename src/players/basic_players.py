#!/usr/bin/env python3


from games.game import Player
import numpy as np


class Human(Player):
    def __init__(self, game):
        self.game = game

    def get_action_id_to_take(self, state, past_action_ids_taken):
        print(self.game.visualize(state))
        print(
            "Your opponents took the following actions:\n%s"
            % "\n".join(
                [
                    self.game.action_2_user_output(a)
                    for a in past_action_ids_taken[1 - self.game.nbr_players :]
                ]
            )
        )
        return self.game.user_input_2_action()

    def process_final_state(self, state, past_action_ids_taken):
        print(self.game.visualize(state))
        points = self.game.get_points(state)
        print("These are the final points:\n%s" % str(points))


class Random(Player):
    def __init__(self, game):
        self.game = game

    def get_action_id_to_take(self, state, past_action_ids_taken):
        return np.random.choice(self.game.get_action_list(state))


class SuperRandom(Player):
    def __init__(self, game):
        self.game = game

    def get_action_id_to_take(self, state, past_action_ids_taken):
        return np.random.choice(self.game.max_nbr_actions)
