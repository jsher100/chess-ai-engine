import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 10000,
}

CENTER_SQUARES = (chess.D4, chess.D5, chess.E4, chess.E5)
MOP_UP_MATERIAL_THRESHOLD = 5


def _manhattan_distance(square_a: int, square_b: int) -> int:
    return abs(chess.square_file(square_a) - chess.square_file(square_b)) + abs(
        chess.square_rank(square_a) - chess.square_rank(square_b)
    )


def _distance_from_center(square: int) -> int:
    return min(_manhattan_distance(square, center) for center in CENTER_SQUARES)


def _king_mobility(board: chess.Board, color: chess.Color) -> int:
    """Number of legal destination squares for `color`'s king."""
    king_square = board.king(color)
    probe = board.copy(stack=False)
    probe.turn = color
    return sum(1 for move in probe.legal_moves if move.from_square == king_square)


def _has_mating_material(board: chess.Board, color: chess.Color) -> bool:
    """True if `color` has a queen, rook, or minor piece left."""
    return any(
        board.pieces(piece_type, color)
        for piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
    )


def _mop_up_applies(board: chess.Board, material: int) -> bool:
    """
    The mop-up term only kicks in once one side is up >= 5 material and the
    losing side has been reduced to just king + pawns (no queen, rook, or
    minor piece left to complicate the mating process).
    """
    if abs(material) < MOP_UP_MATERIAL_THRESHOLD:
        return False
    loser = chess.BLACK if material > 0 else chess.WHITE
    return not _has_mating_material(board, loser)


def _mop_up_bonus(board: chess.Board, material: int) -> int:
    """
    Endgame mop-up term: reward driving the enemy king toward the edge,
    bringing the kings together, and restricting the enemy king's
    mobility -- the three ingredients of a basic king+material mate.
    Only call when _mop_up_applies(board, material) is True.
    """
    winner = chess.WHITE if material > 0 else chess.BLACK
    loser = not winner

    enemy_king = board.king(loser)
    winner_king = board.king(winner)

    center_distance = _distance_from_center(enemy_king)
    king_distance = _manhattan_distance(enemy_king, winner_king)
    enemy_mobility = _king_mobility(board, loser)

    bonus = (
        10 * center_distance
        + 4 * (14 - king_distance)
        + 5 * (8 - enemy_mobility)
    )

    return bonus if winner == chess.WHITE else -bonus


def eval(board: chess.Board) -> int:
    """
    Evaluate material balance from White's perspective: sum(White) - sum(Black).

    Values:
        P = 1, N = 3, B = 3, R = 5, Q = 9, K = 10000
    Uses board.piece_map() from python-chess.

    When one side is up >= 5 material and the losing side has been reduced
    to king + pawns only, an endgame mop-up bonus is added in the winning
    side's favor, and the result is scaled (material * 100 + bonus) so
    material stays dominant over the bonus. Outside that condition, the
    plain material sum is returned unscaled.
    """
    material = 0
    for piece in board.piece_map().values():
        val = PIECE_VALUES[piece.piece_type]
        if piece.color == chess.WHITE:
            material += val
        else:
            material -= val

    if _mop_up_applies(board, material):
        return material * 100 + _mop_up_bonus(board, material)
    return material
