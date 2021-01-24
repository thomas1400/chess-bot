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
        if self.type != PieceType.KNIGHT:
            return self.type.name[0] + "," + self.color.name[0]
        else:
            return "N" + "," + self.color.name[0]

    def color(self):
        return self.color