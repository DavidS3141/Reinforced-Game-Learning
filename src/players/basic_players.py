#!/usr/bin/env python3

from games.game import Player, MCTS_Player
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
        action_id = self.game.user_input_2_action()
        poss_actions = self.game.get_action_list(state)
        while action_id not in poss_actions:
            print("You did not provide a valid action, please try again!")
            action_id = self.game.user_input_2_action()
        return action_id

    def process_final_state(self, state, past_action_ids_taken):
        print(self.game.visualize(state))
        points = self.game.get_points(state)
        print("These are the final points:\n%s" % str(points))


class RandomHuman(Player):
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
        automated_action = np.random.choice(self.game.get_action_list(state))
        print(
            "You choose randomly: %s" % self.game.action_2_user_output(automated_action)
        )
        return automated_action

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


class MCTS_RandomPolicy(MCTS_Player):
    def policy_value_function(self, flatten_state):
        probs = [1.0 / self.mcts.game.max_nbr_actions] * self.mcts.game.max_nbr_actions
        values = [1.0 / self.mcts.game.nbr_players] * self.mcts.game.nbr_players
        return probs, values


class MCTS_NetworkPolicy(MCTS_Player):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def policy_value_function(self, flatten_state):
        output = self.model.predict(flatten_state[None, ...], batch_size=1)
        policy, value = output[:2]
        return policy[0], value[0]
