from typing import List, Optional, Tuple
import chess
from engine.eval import eval, PIECE_VALUES


def minimax(
    board: chess.Board, depth: int, node_count: Optional[List[int]] = None
) -> Tuple[int, Optional[chess.Move]]:
    """
    Plain minimax search without pruning.
    White maximizes, Black minimizes.

    Terminal test is performed first:
    - If checkmate: returns -10000 if White is to move, else +10000.
    - If stalemate or other draw: returns 0.

    eval(board) is only called when depth == 0 and position is not terminal.

    If node_count is given (a single-element list used as a mutable counter),
    it is incremented once for every position visited.
    """
    if node_count is not None:
        node_count[0] += 1

    # Terminal test first
    if board.is_checkmate():
        return (-10000 if board.turn == chess.WHITE else 10000, None)

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_draw()
        or board.is_game_over()
    ):
        return (0, None)

    # Base case: only call eval when depth == 0 and position is not terminal
    if depth == 0:
        return (eval(board), None)

    if board.turn == chess.WHITE:
        best_score = float("-inf")
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            score, _ = minimax(board, depth - 1, node_count)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
        return (int(best_score), best_move)
    else:
        best_score = float("inf")
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            score, _ = minimax(board, depth - 1, node_count)
            board.pop()
            if score < best_score:
                best_score = score
                best_move = move
        return (int(best_score), best_move)


def order_moves(board: chess.Board) -> List[chess.Move]:
    """
    Order legal moves for alpha-beta: captures first (sorted by MVV-LVA —
    most valuable victim first, ties broken by least valuable attacker
    first), then all non-captures.
    """

    def mvv_lva_key(move: chess.Move) -> Tuple[int, int]:
        if board.is_en_passant(move):
            victim_value = PIECE_VALUES[chess.PAWN]
        else:
            victim = board.piece_at(move.to_square)
            victim_value = PIECE_VALUES[victim.piece_type] if victim else 0
        attacker = board.piece_at(move.from_square)
        attacker_value = PIECE_VALUES[attacker.piece_type]
        return (-victim_value, attacker_value)

    captures = [move for move in board.legal_moves if board.is_capture(move)]
    non_captures = [move for move in board.legal_moves if not board.is_capture(move)]
    captures.sort(key=mvv_lva_key)
    return captures + non_captures


def quiescence(
    board: chess.Board,
    alpha: float,
    beta: float,
    node_count: Optional[List[int]] = None,
) -> int:
    """
    Quiescence search: extends the search past the depth horizon along
    capture sequences only, so alphabeta doesn't stop mid-exchange and
    misreport a piece as "won" right before it gets recaptured.

    Same terminal test as alphabeta. Otherwise, stand_pat = eval(board) is
    used as the starting bound for the side to move (White: alpha =
    max(alpha, stand_pat); Black: beta = min(beta, stand_pat)) — the side
    to move is never forced to capture if simply "standing pat" is best —
    and search cuts off once beta <= alpha. Remaining moves are searched
    MVV-LVA-ordered captures only, recursively, with the same max/min logic
    as alphabeta.
    """
    if node_count is not None:
        node_count[0] += 1

    if board.is_checkmate():
        return -10000 if board.turn == chess.WHITE else 10000

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_draw()
        or board.is_game_over()
    ):
        return 0

    stand_pat = eval(board)
    moves = order_moves(board)

    if board.turn == chess.WHITE:
        alpha = max(alpha, stand_pat)
        if beta <= alpha:
            return stand_pat
        best_score = stand_pat
        for move in moves:
            if not board.is_capture(move):
                break  # order_moves puts all captures first
            board.push(move)
            score = quiescence(board, alpha, beta, node_count)
            board.pop()
            if score > best_score:
                best_score = score
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score
    else:
        beta = min(beta, stand_pat)
        if beta <= alpha:
            return stand_pat
        best_score = stand_pat
        for move in moves:
            if not board.is_capture(move):
                break  # order_moves puts all captures first
            board.push(move)
            score = quiescence(board, alpha, beta, node_count)
            board.pop()
            if score < best_score:
                best_score = score
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score


def alphabeta(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    node_count: Optional[List[int]] = None,
    ordered: bool = False,
    quiesce: bool = False,
) -> Tuple[int, Optional[chess.Move]]:
    """
    Minimax search with alpha-beta pruning.
    White maximizes, Black minimizes.

    Same terminal test as minimax(). Prunes a branch once beta <= alpha
    (the opponent already has a better alternative elsewhere, so the
    remaining siblings cannot change the outcome).

    If node_count is given (a single-element list used as a mutable counter),
    it is incremented once for every position visited.

    If ordered=True, moves are generated via order_moves() (MVV-LVA captures
    first, then non-captures) instead of board.legal_moves order, so that
    strong moves are searched first and more branches get pruned.

    If quiesce=True, the depth-0 leaf score comes from quiescence() instead
    of a raw eval(board) call, so the search doesn't stop mid-exchange.
    """
    if node_count is not None:
        node_count[0] += 1

    # Terminal test first
    if board.is_checkmate():
        return (-10000 if board.turn == chess.WHITE else 10000, None)

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_draw()
        or board.is_game_over()
    ):
        return (0, None)

    # Base case: only call eval/quiescence when depth == 0 and not terminal
    if depth == 0:
        if quiesce:
            return (quiescence(board, alpha, beta, node_count), None)
        return (eval(board), None)

    moves = order_moves(board) if ordered else board.legal_moves

    if board.turn == chess.WHITE:
        best_score = float("-inf")
        best_move = None
        for move in moves:
            board.push(move)
            score, _ = alphabeta(
                board, depth - 1, alpha, beta, node_count, ordered, quiesce
            )
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return (int(best_score), best_move)
    else:
        best_score = float("inf")
        best_move = None
        for move in moves:
            board.push(move)
            score, _ = alphabeta(
                board, depth - 1, alpha, beta, node_count, ordered, quiesce
            )
            board.pop()
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return (int(best_score), best_move)
