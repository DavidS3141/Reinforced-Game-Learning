#!/usr/bin/env python3

from abc import ABC, abstractmethod


class GameEngine:
    def __init__(self, game, list_of_players):
        assert len(list_of_players) == game.nbr_players
        self.game = game
        self.players = list_of_players

    def run(self):
        state = self.game.generate_initial_state()
        past_action_ids_taken = []
        while not self.game.is_terminal(state):
            player_id = self.game.get_player_turn(state)
            player = self.players[player_id]
            action_id = player.get_action_id_to_take(state, past_action_ids_taken)
            past_action_ids_taken.append(action_id)
            state = self.game.take_action(state, player_id, action_id)
        for player in self.players:
            player.process_final_state(state, past_action_ids_taken)
        return self.game.get_points(state)


class Game(ABC):
    @property
    @abstractmethod
    def nbr_players(self):
        return 2

    @property
    @abstractmethod
    def state_dimensionality(self):
        return 27

    @property
    @abstractmethod
    def max_nbr_actions(self):
        return 9

    @abstractmethod
    def generate_initial_state(self):
        return "initial state representation"

    @abstractmethod
    def flatten_state(self, state):
        """
        returns a 1D numpy array with the size equal to state dimensionality
        The representation in this case should be invariant to the player id
        and from the variant to the current players turn.
        """
        pass

    @abstractmethod
    def visualize(self, state):
        """
        some string that can be dumped to the terminal,
        should be invariant to players turn and variant to player id.
        """
        pass

    @abstractmethod
    def get_player_turn(self, state):
        return "integer player id from 0 to nbr_players"

    @abstractmethod
    def take_action(self, state, player_id, action_id):
        return "Game State"

    @abstractmethod
    def is_terminal(self, state):
        return "bool"

    @abstractmethod
    def get_points(self, state):
        assert self.is_terminal(state)
        return "list of points for every player"

    @abstractmethod
    def get_action_list(self, state):
        """
        list of action ids between 0 to max_nbr_actions of actions that are allowed
        """
        pass

    @abstractmethod
    def user_input_2_action(self):
        return "some action id based on some terminal user input"

    @abstractmethod
    def action_2_user_output(self, action):
        return "some string that decodes an action id into a human readable action"


class Player(ABC):
    @abstractmethod
    def get_action_id_to_take(self, state, past_action_ids_taken):
        return "action id that this player wants to take"

    def process_final_state(self, state, past_action_ids_taken):
        pass
