
import numpy as np
from game.piece import Piece

class Board:
    """
    Represents the Quarto game board and its state.
    """
    def __init__(self):
        self.board = np.full((4, 4), None)
        self.available_pieces = set(range(16))
        self.current_player = 0

    def place_piece(self, piece_id, row, col):
        """
        Places a piece on the board.
        """
        if not self.is_valid_placement(piece_id, row, col):
            raise ValueError("Invalid placement.")
        
        self.board[row, col] = Piece(piece_id)

    def is_valid_placement(self, piece_id, row, col):
        """
        Checks if a placement is valid.
        """
        return (0 <= row < 4 and 0 <= col < 4 and
                self.board[row, col] is None)

    def check_win(self):
        """
        Checks for a winning condition on the board.
        """
        # Check rows
        for i in range(4):
            if Piece.check_line_for_win(self.board[i, :]):
                return True
        
        # Check columns
        for i in range(4):
            if Piece.check_line_for_win(self.board[:, i]):
                return True

        # Check diagonals
        if Piece.check_line_for_win(np.diag(self.board)):
            return True
        if Piece.check_line_for_win(np.diag(np.fliplr(self.board))):
            return True
            
        return False

    def is_full(self):
        """
        Checks if the board is full.
        """
        return np.all(self.board != None)

    def get_state(self):
        """
        Returns the current state of the board as a numpy array.
        The state includes the board configuration and the next piece to be placed.
        """
        # Represent the board state as a 4x4 grid of piece_ids, with -1 for empty cells.
        board_state = np.full((4, 4), -1)
        for r in range(4):
            for c in range(4):
                if self.board[r, c] is not None:
                    board_state[r, c] = self.board[r, c].piece_id
        return board_state

    def reset(self):
        """
        Resets the board to its initial state.
        """
        self.__init__()
