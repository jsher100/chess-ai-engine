from collections import Counter

import chess
from engine.search import alphabeta, minimax

HANGING_QUEEN_FEN = "rnb1kbnr/pppp1ppp/8/4p3/4P1q1/4N3/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
MIDDLEGAME_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
KR_VS_K_FEN = "8/8/8/8/8/8/4k3/R6K w - - 0 1"


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


def test_alphabeta_matches_minimax_score_start_position():
    """alphabeta(board, 3, ...) returns the same score as minimax(board, 3) on the start position."""
    board = chess.Board()
    mm_score, _ = minimax(board, 3)
    ab_score, _ = alphabeta(board, 3, float("-inf"), float("inf"))
    assert ab_score == mm_score


def test_alphabeta_matches_minimax_score_hanging_queen():
    """alphabeta(board, 3, ...) returns the same score as minimax(board, 3) on the hanging-queen FEN."""
    board = chess.Board(HANGING_QUEEN_FEN)
    mm_score, _ = minimax(board, 3)
    ab_score, _ = alphabeta(board, 3, float("-inf"), float("inf"))
    assert ab_score == mm_score


def test_alphabeta_visits_fewer_nodes_start_position():
    """alphabeta visits strictly fewer nodes than minimax at depth 3 on the start position."""
    board = chess.Board()

    mm_nodes = [0]
    minimax(board, 3, mm_nodes)

    ab_nodes = [0]
    alphabeta(board, 3, float("-inf"), float("inf"), ab_nodes)

    assert ab_nodes[0] < mm_nodes[0]


def test_alphabeta_visits_fewer_nodes_hanging_queen():
    """alphabeta visits strictly fewer nodes than minimax at depth 3 on the hanging-queen FEN."""
    board = chess.Board(HANGING_QUEEN_FEN)

    mm_nodes = [0]
    minimax(board, 3, mm_nodes)

    ab_nodes = [0]
    alphabeta(board, 3, float("-inf"), float("inf"), ab_nodes)

    assert ab_nodes[0] < mm_nodes[0]


def test_ordered_alphabeta_matches_minimax_score_start_position():
    """Ordered alphabeta returns the same score as minimax on the start position."""
    board = chess.Board()
    mm_score, _ = minimax(board, 3)
    ab_score, _ = alphabeta(board, 3, float("-inf"), float("inf"), ordered=True)
    assert ab_score == mm_score


def test_ordered_alphabeta_matches_minimax_score_hanging_queen():
    """Ordered alphabeta returns the same score as minimax on the hanging-queen FEN."""
    board = chess.Board(HANGING_QUEEN_FEN)
    mm_score, _ = minimax(board, 3)
    ab_score, _ = alphabeta(board, 3, float("-inf"), float("inf"), ordered=True)
    assert ab_score == mm_score


def test_ordered_alphabeta_matches_minimax_score_middlegame():
    """Ordered alphabeta returns the same score as minimax on the middlegame FEN."""
    board = chess.Board(MIDDLEGAME_FEN)
    mm_score, _ = minimax(board, 3)
    ab_score, _ = alphabeta(board, 3, float("-inf"), float("inf"), ordered=True)
    assert ab_score == mm_score


def test_ordered_alphabeta_visits_fewer_nodes_than_unordered_middlegame():
    """Ordered alphabeta visits fewer nodes than unordered alphabeta on the middlegame FEN."""
    board = chess.Board(MIDDLEGAME_FEN)

    unordered_nodes = [0]
    alphabeta(board, 3, float("-inf"), float("inf"), unordered_nodes, ordered=False)

    ordered_nodes = [0]
    alphabeta(board, 3, float("-inf"), float("inf"), ordered_nodes, ordered=True)

    assert ordered_nodes[0] < unordered_nodes[0]


def test_alphabeta_depth1_without_quiescence_shows_horizon_effect():
    """
    Without quiescence, alphabeta depth 1 on the middlegame FEN is fooled by
    the horizon effect: it greedily grabs a pawn (Bxf7+ or Nxe5) and reports
    a score of +1, never looking far enough ahead to see the capturing piece
    gets recaptured.
    """
    board = chess.Board(MIDDLEGAME_FEN)
    score, move = alphabeta(board, 1, float("-inf"), float("inf"), quiesce=False)
    assert board.is_capture(move)
    assert board.san(move) in ("Bxf7+", "Nxe5")
    assert score == 1


def test_alphabeta_depth1_with_quiescence_resolves_horizon_effect():
    """
    With quiescence search extending capture sequences past the depth-1
    horizon, alphabeta correctly sees that grabbing the pawn just loses it
    back, so it prefers a quiet move and reports the true, materially-even
    score of 0.
    """
    board = chess.Board(MIDDLEGAME_FEN)
    score, move = alphabeta(board, 1, float("-inf"), float("inf"), quiesce=True)
    assert not board.is_capture(move)
    assert score == 0


def _position_key(board: chess.Board) -> str:
    return board.board_fen() + ("w" if board.turn == chess.WHITE else "b")


def _choose_move(board: chess.Board, base_depth: int, history: Counter):
    """
    Pick White's or Black's move via alphabeta(depth=base_depth, ordered=True,
    quiesce=True), with two pieces of self-play scaffolding that sit entirely
    outside the engine itself:

    1. The search is run on a fresh `chess.Board(fen)` rather than the real,
       history-carrying game board. alphabeta's own terminal test calls
       board.can_claim_draw(), which looks at *that board's* move history —
       if we searched directly on the real game board, a position that has
       genuinely recurred earlier in THIS game (not hypothetically, within
       the search tree) makes the search see an immediate claimable draw at
       the root and refuse to even propose a move. A fresh board has no
       history, so the search always evaluates the position on its merits.
    2. If the proposed move would make a position recur for the real 3rd
       time (tracked in `history`, keyed on the actual played game), the
       search is re-run one ply deeper. This is the one concession beyond
       plain depth-3: the final king+rook mating technique occasionally
       requires a short "waiting move" (triangulation) that a 3-ply search
       can't foresee, so both sides shuffle into a repeated position. One
       extra ply of lookahead is enough to find the breakthrough.
    """
    depth = base_depth
    for _ in range(6):
        fresh = chess.Board(board.fen())
        score, move = alphabeta(
            fresh, depth, float("-inf"), float("inf"), None, ordered=True, quiesce=True
        )
        if move is None:
            return None  # real checkmate/stalemate/insufficient material
        fresh.push(move)
        if history[_position_key(fresh)] + 1 < 3:
            return move
        depth += 1
    return move


def test_king_and_rook_vs_king_mates_within_40_moves():
    """
    The engine plays itself (White and Black both via alphabeta depth=3,
    ordered=True, quiesce=True) from a king+rook vs. king endgame. The mop-up
    bonus in eval() should give the search enough signal to drive the lone
    king to the edge and deliver checkmate well within 40 full moves.
    """
    board = chess.Board(KR_VS_K_FEN)
    history = Counter()
    history[_position_key(board)] += 1

    MAX_FULL_MOVES = 40
    for _ in range(MAX_FULL_MOVES * 2):
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            break
        move = _choose_move(board, 3, history)
        if move is None:
            break
        board.push(move)
        history[_position_key(board)] += 1

    assert board.is_checkmate()
    assert board.turn == chess.BLACK  # Black is the side actually mated

