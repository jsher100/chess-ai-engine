import time

import chess

from engine.search import alphabeta, minimax


def main() -> None:
    depth = 3
    board = chess.Board()

    mm_nodes = [0]
    start = time.perf_counter()
    mm_score, mm_move = minimax(board, depth, mm_nodes)
    mm_time = time.perf_counter() - start

    ab_nodes = [0]
    start = time.perf_counter()
    ab_score, ab_move = alphabeta(board, depth, float("-inf"), float("inf"), ab_nodes)
    ab_time = time.perf_counter() - start

    print(f"Start position, depth {depth}")
    print(f"  minimax:   time={mm_time:.4f}s  nodes={mm_nodes[0]}  score={mm_score}  move={mm_move}")
    print(f"  alphabeta: time={ab_time:.4f}s  nodes={ab_nodes[0]}  score={ab_score}  move={ab_move}")


if __name__ == "__main__":
    main()
