import time

import chess

from engine.search import alphabeta, minimax

MIDDLEGAME_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


def bench_minimax(board: chess.Board, depth: int):
    nodes = [0]
    start = time.perf_counter()
    score, move = minimax(board, depth, nodes)
    elapsed = time.perf_counter() - start
    return score, move, nodes[0], elapsed


def bench_alphabeta(board: chess.Board, depth: int, ordered: bool):
    nodes = [0]
    start = time.perf_counter()
    score, move = alphabeta(
        board, depth, float("-inf"), float("inf"), nodes, ordered
    )
    elapsed = time.perf_counter() - start
    return score, move, nodes[0], elapsed


def report(label: str, score, move, nodes: int, elapsed: float) -> None:
    print(f"  {label:<22} time={elapsed:.4f}s  nodes={nodes}  score={score}  move={move}")


def main() -> None:
    depth = 3

    print(f"Start position, depth {depth}")
    board = chess.Board()
    report("minimax:", *bench_minimax(board, depth))
    report("alphabeta:", *bench_alphabeta(board, depth, ordered=False))

    print(f"\nMiddlegame position, depth {depth}")
    board = chess.Board(MIDDLEGAME_FEN)
    report("minimax:", *bench_minimax(board, depth))
    report("alphabeta (unordered):", *bench_alphabeta(board, depth, ordered=False))
    report("alphabeta (ordered):", *bench_alphabeta(board, depth, ordered=True))


if __name__ == "__main__":
    main()
