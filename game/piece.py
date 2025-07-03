class Piece:
    """
    Represents a Quarto piece.

    Each piece has 4 attributes, represented by a 4-bit integer.
    - Bit 3: Color (0: Light, 1: Dark)
    - Bit 2: Height (0: Short, 1: Tall)
    - Bit 1: Shape (0: Round, 1: Square)
    - Bit 0: Hollowness (0: Solid, 1: Hollow)
    """
    def __init__(self, piece_id):
        if not 0 <= piece_id < 16:
            raise ValueError("piece_id must be between 0 and 15.")
        self.piece_id = piece_id
        self.attributes = format(piece_id, '04b')

    def __repr__(self):
        return f"Piece({self.piece_id}, {self.attributes})"

    def has_common_attribute(self, other_piece):
        """
        Checks if two pieces share at least one common attribute.
        This is determined by a bitwise OR or AND operation.
        If the bitwise AND is not zero, they share a "1" attribute.
        If the bitwise OR of their inverted bits is not 15, they share a "0" attribute.
        """
        return (self.piece_id & other_piece.piece_id) != 0 or \
               (~self.piece_id & ~other_piece.piece_id & 0b1111) != 0

    @staticmethod
    def check_line_for_win(pieces):
        """
        Checks if a line of 4 pieces has a winning combination.
        A line wins if all 4 pieces share at least one common attribute.
        """
        if None in pieces or len(pieces) != 4:
            return False

        # Bitwise AND of all piece IDs. If > 0, all have a common "1" attribute.
        common_and = pieces[0].piece_id & pieces[1].piece_id & pieces[2].piece_id & pieces[3].piece_id
        if common_and > 0:
            return True

        # Bitwise OR of all piece IDs. If < 15, all have a common "0" attribute.
        common_or = pieces[0].piece_id | pieces[1].piece_id | pieces[2].piece_id | pieces[3].piece_id
        if common_or < 15:
            return True

        return False
