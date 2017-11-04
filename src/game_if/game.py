#!/user/bin/env python

'''game.py: Provides the game class interface for implementing full information games.'''

################################################################################

class Game:

    nbr_players = 2
    state_dim = 27

    def __init__(self):
        return

    def visualize():
        return

    def get_player_turn():
        return

    def take_action(player_id, action_id):
        return

    def get_status():
        return player_id_who_has_won #(-1 if still ongoing, -2 if draw, -3 if invalid move)

    def get_action_list():
        return

    def get_state_for_player(player_id):
        return
