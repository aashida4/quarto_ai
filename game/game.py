from game.board import Board

class QuartoGame:
    """
    Manages the overall game flow of Quarto.
    """
    def __init__(self):
        self.board = Board()
        self.current_player = 0  # Player 0 or 1
        self.piece_to_place = None
        self.game_over = False
        self.winner = None

    def select_piece(self, piece_id):
        """
        A player selects a piece for the opponent to place.
        """
        if piece_id not in self.board.available_pieces:
            raise ValueError("Selected piece is not available.")
        self.piece_to_place = piece_id
        self.board.available_pieces.remove(piece_id)
        return True

    def place_piece(self, row, col):
        """
        A player places the piece selected by the opponent.
        """
        if self.piece_to_place is None:
            raise ValueError("No piece has been selected to be placed.")

        if not self.board.is_valid_placement(self.piece_to_place, row, col):
            return False # Invalid move

        self.board.place_piece(self.piece_to_place, row, col)

        if self.board.check_win():
            self.game_over = True
            self.winner = self.current_player
        elif self.board.is_full():
            self.game_over = True # Draw

        self.piece_to_place = None
        return True

    def switch_player(self):
        """
        Switches the current player.
        """
        self.current_player = 1 - self.current_player

    def get_state(self):
        """
        Gets the combined state of the game for the AI agent.
        The state includes the board configuration, the next piece to be placed,
        the current player, and the current game phase (placement or selection).
        """
        board_state = self.board.get_state().flatten()
        piece_to_place_state = self.piece_to_place / 15.0 if self.piece_to_place is not None else -1.0
        
        # Determine game phase: 0 for placement, 1 for selection
        game_phase = 0 if self.piece_to_place is not None else 1

        return tuple(board_state.tolist() + [piece_to_place_state, self.current_player, game_phase])


    def reset(self):
        """
        Resets the game to its initial state.
        """
        self.__init__()

    def is_game_over(self):
        return self.game_over