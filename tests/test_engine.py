import chess
from engine.search import minimax


def test_start_position():
    """From chess.Board() (start position), minimax(board, 3) returns a legal move."""
    board = chess.Board()
    score, move = minimax(board, 3)
    assert move is not None
    assert move in board.legal_moves


def test_hanging_queen():
    """
    Test that minimax(board, 3) capitalizes on a hanging queen.

    Note on FEN coordinates:
    The prompt specifies:
      FEN: "rnb1kbnr/pppp1ppp/8/4p3/4P1q1/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
      "(Black's queen on g4 is hanging to the knight on f3), minimax(board, 3) returns the move Nxg4."

    In standard chess rules, a knight on f3 cannot attack or move to g4 (a knight moves
    in an L-shape; f3 to g4 is a diagonal step). In that exact position, f3g4 is not a legal
    move in python-chess, and minimax returns a legal move (such as Qe2).

    In the intended tactical position where Black's queen on g4 is hanging to a White knight
    and captured via 'Nxg4', the knight is on e3:
      FEN: "rnb1kbnr/pppp1ppp/8/4p3/4P1q1/4N3/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
    """
    # 1. Corrected tactical position: knight on e3 captures hanging queen on g4 (Nxg4)
    board_hanging_queen = chess.Board(
        "rnb1kbnr/pppp1ppp/8/4p3/4P1q1/4N3/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
    )
    score, move = minimax(board_hanging_queen, 3)
    assert move is not None
    assert board_hanging_queen.san(move) == "Nxg4"

    # 2. Also verify on prompt's literal FEN that minimax returns a valid legal move
    board_prompt = chess.Board(
        "rnb1kbnr/pppp1ppp/8/4p3/4P1q1/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
    )
    score, move_prompt = minimax(board_prompt, 3)
    assert move_prompt is not None
    assert move_prompt in board_prompt.legal_moves

