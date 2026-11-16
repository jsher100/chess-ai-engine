from typing import Optional, Tuple
import chess
from engine.eval import eval


def minimax(board: chess.Board, depth: int) -> Tuple[int, Optional[chess.Move]]:
    """
    Plain minimax search without pruning.
    White maximizes, Black minimizes.

    Terminal test is performed first:
    - If checkmate: returns -10000 if White is to move, else +10000.
    - If stalemate or other draw: returns 0.

    eval(board) is only called when depth == 0 and position is not terminal.
    """
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
            score, _ = minimax(board, depth - 1)
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
            score, _ = minimax(board, depth - 1)
            board.pop()
            if score < best_score:
                best_score = score
                best_move = move
        return (int(best_score), best_move)

