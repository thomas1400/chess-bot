from .ChessPiece import ChessPiece, PieceColor, PieceType

CHESSBOARD_HOME_RANK = [
    PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN, 
    PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK
    ]

class ChessBoard:

    def __init__(self):
        self.board = [[] for i in range(8)]
        for type in CHESSBOARD_HOME_RANK:
            self.board[0].append(ChessPiece(type, PieceColor.WHITE))
        for i in range(8):
            self.board[1].append(ChessPiece(PieceType.PAWN, PieceColor.WHITE))

        for i in range(8):
            self.board[6].append(ChessPiece(PieceType.PAWN, PieceColor.BLACK))
        for type in CHESSBOARD_HOME_RANK:
            self.board[7].append(ChessPiece(type, PieceColor.BLACK))
    
    def __str__(self):
        res = ""
        for rank in reversed(self.board):
            for piece in rank:
                res += str(piece) + "\t"
            res += "\n"
        return res
    
    def move(from, to):
        pass
        
            
