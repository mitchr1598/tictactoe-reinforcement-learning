import numpy as np
from game import Game
from dataclasses import dataclass


@dataclass()
class GameInfo:
    p1_board: np.array
    p2_board: np.array
    player: int
    move_chosen: int
    move_predictions: np.array
    win: bool = None


@dataclass
class GameResults:
    board_inputs: np.array
    move_predictions: np.array
    move_chosen: np.array
    win: np.array

    @property
    def prediction_adjustment(self):
        pa = []
        for mp, mc, w in zip(self.move_predictions, self.move_chosen, self.win):
            if w:
                pa
                pa.append(np.zeros(mp.shape))
        return


class DataGenerator:
    def __init__(self, network):
        self.network = network
        self.results = []
        self.x, self.y = None, None
    
    def sim_games(self, n):
        self.results = [self.run_game() for _ in range(n)]
        self.x = np.stack([d1 for d1, _, _ in self.results], axis=0)  # Each board in each game sim for each turn
        self.y = np.stack([(d2 * d3) / (d2 * d3).sum() for _, d2, d3 in self.results], axis=0)  # Each network prediction, with
        #  self.y NEEDS WORK. WE'RE GETTING THE MOVE PREDICTIONS ARE BUT ARE PROPERLY ZEROING THE WINS/LOSSESS

    def run_game(self):
        g = Game()
        new_data = []
        while not g.game_finished():
            b1, b2 = g.get_boards()
            player = g.players_turn()
            valid_moves = g.valid_moves()  # Flattened Boolean array of length 9
            best_move, predictions = self.network.choose_move(b1, b2, valid_moves)
            new_data.append(GameInfo(b1, b2, player, best_move, predictions))
            g.make_move_skip_validation(player, best_move)
        winner = 1 if g.is_winner()[0] else 2
        for gi in new_data:
            gi.win = gi.player == winner
        self.results.append(
            GameResults(
                np.array([np.concatenate((gi.p1_board, gi.p2_board), axis=1) for gi in new_data]),  # The new x data
                np.array([gi.move_predictions for gi in new_data]),  # the moves guessed at, to be used in y data construction
                np.array([gi.move_chosen for gi in new_data]),
                np.array([gi.win for gi in new_data])  # boolean on if won, to be used in y data construction
            )
        )
        




