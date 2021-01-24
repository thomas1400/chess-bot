from .ChessPiece import ChessPiece, PieceColor, PieceType

CHESSBOARD_HOME_RANK = [
    PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN, 
    PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK
    ]

class ChessBoard:

    def __init__(self):
        self.board = [[] for i in range(8)]
        self.white_turn = True
        for type in CHESSBOARD_HOME_RANK:
            self.board[0].append(ChessPiece(type, PieceColor.WHITE))
        for i in range(8):
            self.board[1].append(ChessPiece(PieceType.PAWN, PieceColor.WHITE))
        
        for rank in range(2, 6):
            for i in range(8):
                self.board[rank].append(None)

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
    
    def get_turn_color(self):
        if self.white_turn:
            return PieceColor.WHITE
        else:
            return PieceColor.BLACK

    def get_square(self, file, rank):
        return self.board[rank][file]
    
    def move(self, move_from, move_to):
        self.board[move_to[1]][move_to[0]] = self.board[move_from[1]][move_from[0]]
        self.board[move_from[1]][move_from[0]] = None

    def interpret_move(self, squares):
        # squares are passed in as [F, R], file (0-7 representing A-H) + rank (0-7 representing 1-8)

        # for a normal move
        if len(squares) == 2:
            pieces = []

            for square in squares:
                piece = self.get_square(square[0], square[1])
                pieces.append(piece)
            
            if pieces[0] is not None and pieces[1] is None:
                self.move(squares[0], squares[1])
            elif pieces[1] is not None and pieces[0] is None:
                self.move(squares[1], squares[0])
            elif pieces[0] is not None and pieces[1] is not None:
                # this is a piece taking another
                # the piece whose turn it is will take the other
                turn_color = self.get_turn_color()
                for i in range(len(pieces)):
                    if pieces[i].color() == turn_color:
                        self.move(squares[i], squares[1 - i])
        
        # for castling
        elif len(squares) == 4:
            # possible castle squares are:
            # White king side:  E1, F1, G1, H1, E1->G1, H1->F1
            wk_castle = [4, 0], [5, 0], [6, 0], [7, 0]
            # White queen side: A1, C1, D1, E1 -> 
            wq_castle = [0, 0], [2, 0], [3, 0], [4, 0]
            # Black king side:  E8, F8, G8, H8 -> 
            bk_castle = [4, 7], [5, 7], [6, 7], [7, 7]
            # Black queen side: A8, C8, D8, E8 -> 
            bq_castle = [0, 7], [2, 7], [3, 7], [4, 7]

            squares.sort()
            if squares == wk_castle or squares == bk_castle:
                self.move(squares[0], squares[2])
                self.move(squares[3], squares[1])
            elif squares == wq_castle or squares == wq_castle:
                self.move(squares[3], squares[1])
                self.move(squares[0], squares[2])




        # TODO: add a check for pawn promotion

        # TODO: add a check for en passant
        
        return False


        # five types of moves:
        # 1. piece moving to an empty square
        #   two squares given, one empty and one not
        # 2. piece taking a piece of the opposite color
        #   two squares given, the piece whose turn it is takes the other
        # 3. castling
        #   four squares given, brute force check this one
        # 4. promoting a pawn -- automatically promote to queen
        #   two squares given, specifically have to check for pawn movement on opposite home row
        # 5. en passant - requires change in 3 squares, more complicated checks

