from enum import Enum, auto

class PieceType(Enum):
    KING = auto()
    QUEEN = auto()
    ROOK = auto()
    KNIGHT = auto()
    BISHOP = auto()
    PAWN = auto()

class PieceColor(Enum):
    WHITE = auto()
    BLACK = auto()

class ChessPiece:
    def __init__(self, type, color, hasMoved=False):
        self.type = type
        self.color = color
        self.hasMoved = hasMoved

    def __str__(self):
        return self.type.name[0:2] + "," + self.color.name[0]