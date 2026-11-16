import time
from typing import Optional

import chess
from flask import Flask, jsonify, render_template, request

from engine.search import alphabeta

app = Flask(__name__)

SEARCH_DEPTH = 3


def describe_game_over(board: chess.Board) -> Optional[str]:
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate — {winner} wins!"
    if board.is_stalemate():
        return "Draw by stalemate."
    if board.is_insufficient_material():
        return "Draw by insufficient material."
    if board.can_claim_draw():
        return "Draw (fifty-move rule or repetition)."
    if board.is_game_over():
        return "Game over."
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/move", methods=["POST"])
def move():
    data = request.get_json(force=True, silent=True) or {}
    fen = data.get("fen")
    move_uci = data.get("move_uci")

    if not fen or not move_uci:
        return jsonify({"error": "fen and move_uci are required"}), 400

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "invalid fen"}), 400

    if board.turn != chess.WHITE:
        return jsonify({"error": "it is not White's turn in the given position"}), 400

    try:
        human_move = chess.Move.from_uci(move_uci)
    except ValueError:
        return jsonify({"error": "invalid move_uci"}), 400

    if human_move not in board.legal_moves:
        return jsonify({"error": "illegal move"}), 400

    board.push(human_move)

    if board.is_game_over(claim_draw=True):
        return jsonify(
            {
                "fen": board.fen(),
                "engine_move": None,
                "score": None,
                "nodes": 0,
                "time_ms": 0,
                "game_over": True,
                "message": describe_game_over(board),
            }
        )

    node_count = [0]
    start = time.perf_counter()
    score, engine_move = alphabeta(
        board,
        SEARCH_DEPTH,
        float("-inf"),
        float("inf"),
        node_count,
        ordered=True,
        quiesce=True,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    engine_move_uci = engine_move.uci() if engine_move is not None else None
    if engine_move is not None:
        board.push(engine_move)

    return jsonify(
        {
            "fen": board.fen(),
            "engine_move": engine_move_uci,
            "score": score,
            "nodes": node_count[0],
            "time_ms": round(elapsed_ms, 1),
            "game_over": board.is_game_over(claim_draw=True),
            "message": describe_game_over(board),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
