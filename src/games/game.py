#!/usr/bin/env python3

from time import sleep
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import os.path as osp
from algorithm.mcts import MCTS
from utils import get_time_stamp
from tools.write_parse_tfrecords import write_tfrecord


class GameEngine:
    def __init__(self, game, list_of_players):
        assert len(list_of_players) == game.nbr_players
        self.game = game
        self.players = list_of_players

    def run(self):
        state = self.game.generate_initial_state()
        past_action_ids_taken = []
        with tqdm() as pbar:
            while not self.game.is_terminal(state):
                player_id = self.game.get_player_turn(state)
                player = self.players[player_id]
                action_id = player.get_action_id_to_take(state, past_action_ids_taken)
                past_action_ids_taken.append(action_id)
                state = self.game.take_action(state, player_id, action_id)
                pbar.update(1)
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


class MCTS_Player(Player):
    def __init__(
        self,
        game,
        train_data_output_dir=None,
        temperature=0.0,
        nbr_sims=32,
        c_puct=20.0,
    ):
        self.past_action_ids_taken = None
        self.mcts = MCTS(
            game,
            self.policy_value_function,
            nbr_sims=nbr_sims,
            temperature=temperature,
            c_puct=c_puct,
        )
        self.train_data_output_dir = train_data_output_dir
        if self.train_data_output_dir is not None:
            self.log_game_states = []
            self.log_priors = []

    def get_action_probabilities_state_values(self, state, past_action_ids_taken):
        if self.mcts.root is None:
            if self.train_data_output_dir is not None:
                assert len(past_action_ids_taken) == 0
        else:
            assert self.past_action_ids_taken is not None
            start_idx = len(self.past_action_ids_taken)
            if self.train_data_output_dir is not None:
                assert len(past_action_ids_taken) - start_idx == 1
            for action_id in past_action_ids_taken[start_idx:]:
                self.set_next_action(action_id)
        self.past_action_ids_taken = list(past_action_ids_taken)
        probs, values = self.mcts.evaluate(state)
        if self.train_data_output_dir is not None:
            self.log_game_states.append(state)
            self.log_priors.append(probs)
        print(
            "Estimated value for position and current player: %.1f%%"
            % (np.array(probs) * np.array(values) * 100.0).sum()
        )
        return probs, values

    def get_action_id_to_take(self, state, past_action_ids_taken):
        probs, values = self.get_action_probabilities_state_values(
            state, past_action_ids_taken
        )
        selected_action = np.random.choice(self.mcts.game.max_nbr_actions, p=probs)
        return selected_action

    def set_next_action(self, action_id):
        self.mcts.cut_root(action_id)

    def process_final_state(self, state, past_action_ids_taken):
        self.mcts.reset()
        if self.train_data_output_dir is not None:
            points = list(self.mcts.game.get_points(state))
            timestamp = get_time_stamp()
            for i in range(len(self.log_game_states)):
                s = self.log_game_states[i]
                p = self.log_priors[i]
                turn = self.mcts.game.get_player_turn(s)
                v = points[turn:] + points[:turn]

                data_dict = dict(
                    flatten_state=self.mcts.game.flatten_state(s).astype(np.float32),
                    policy_label=np.array(p).astype(np.float32),
                    values_label=np.array(v).astype(np.float32),
                    turn_nbr=np.array(i).astype(np.int32),
                )
                fixed_dims = dict(
                    flatten_state=(self.mcts.game.state_dimensionality,),
                    policy_label=(self.mcts.game.max_nbr_actions,),
                    values_label=(self.mcts.game.nbr_players,),
                    turn_nbr=(),
                )
                filename = osp.join(
                    self.train_data_output_dir, "%s_%03d" % (timestamp, i)
                )
                while osp.exists(filename + ".tfrecords"):
                    sleep(1)
                    timestamp = get_time_stamp()
                    filename = osp.join(
                        self.train_data_output_dir, "%s_%03d" % (timestamp, i)
                    )
                assert not osp.exists(filename + ".tfrecords")
                write_tfrecord(data_dict, filename, dimension_dict=fixed_dims)
            self.log_game_states = []
            self.log_priors = []

    @abstractmethod
    def policy_value_function(self, flatten_state):
        pass
