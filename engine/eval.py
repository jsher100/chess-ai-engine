import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 10000,
}


def eval(board: chess.Board) -> int:
    """
    Evaluate material balance from White's perspective: sum(White) - sum(Black).

    Values:
        P = 1, N = 3, B = 3, R = 5, Q = 9, K = 10000
    Uses board.piece_map() from python-chess.
    """
    score = 0
    for piece in board.piece_map().values():
        val = PIECE_VALUES[piece.piece_type]
        if piece.color == chess.WHITE:
            score += val
        else:
            score -= val
    return score

