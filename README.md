# Chess AI Engine

A lightweight Python chess engine implementing static material evaluation, depth-limited minimax search, alpha-beta pruning, and MVV-LVA move ordering.

## Features

- **Material Evaluation (`engine/eval.py`)**: Computes static board score from White's perspective ($\sum \text{White} - \sum \text{Black}$) using piece values: Pawn = 1, Knight = 3, Bishop = 3, Rook = 5, Queen = 9, King = 10000.
- **Minimax Search (`engine/search.py`)**: Depth-limited adversarial search where White maximizes and Black minimizes. Terminal states (checkmate $\pm 10000$, draws $0$) are evaluated first.
- **Alpha-Beta Pruning (`engine/search.py`)**: Significantly cuts down the search tree by pruning branches where $\beta \le \alpha$, preserving exact minimax evaluation scores while visiting a fraction of the nodes.
- **MVV-LVA Move Ordering (`engine/search.py`)**: Prioritizes captures using Most Valuable Victim – Least Valuable Attacker ordering, promoting early alpha-beta cutoffs.
- **Benchmarking Suite (`bench.py`)**: Measures execution time, node count, and move selection across search configurations on opening and middlegame positions.

## Project Structure

```text
chess-ai-engine/
├── engine/
│   ├── __init__.py        # Exports eval, minimax, alphabeta, order_moves
│   ├── eval.py            # Static material evaluation
│   └── search.py          # Minimax, alpha-beta search, and MVV-LVA move ordering
├── tests/
│   ├── __init__.py
│   └── test_engine.py     # Test suite verifying search correctness, scores, and pruning
├── bench.py               # Benchmark script comparing minimax vs. alpha-beta
├── requirements.txt       # Project dependencies (python-chess, pytest)
└── README.md
```

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

Run the full test suite using `pytest`:

```bash
pytest -v
```

Or using the virtualenv runner directly:

```bash
.venv/bin/pytest -v
```

The test suite validates:
- Legal move selection from the opening position
- Exploiting tactical blunders (capturing hanging pieces)
- Score parity between minimax and alpha-beta (both unordered and ordered)
- Node count reductions achieved by alpha-beta pruning and MVV-LVA ordering

## Running Benchmarks

Run the benchmark script to compare performance across algorithms:

```bash
python bench.py
```

### Sample Benchmark Results (Depth 3)

| Position | Algorithm | Time (s) | Nodes Visited | Node Reduction |
| :--- | :--- | :--- | :--- | :--- |
| **Start Position** | Minimax | ~1.27s | 9,323 | Baseline |
| | Alpha-Beta (unordered) | ~0.08s | 581 | **~93.8% fewer nodes** |
| **Middlegame** | Minimax | ~5.41s | 31,506 | Baseline |
| | Alpha-Beta (unordered) | ~0.30s | 1,855 | **~94.1% fewer nodes** |
| | Alpha-Beta (MVV-LVA ordered) | ~0.28s | 1,483 | **~95.3% fewer nodes** |

## Quickstart & Usage

```python
import chess
from engine.eval import eval
from engine.search import alphabeta, minimax

board = chess.Board()

# Static evaluation
score = eval(board)
print(f"Material evaluation: {score}")

# Alpha-beta search at depth 3 with move ordering
score, best_move = alphabeta(
    board, depth=3, alpha=float("-inf"), beta=float("inf"), ordered=True
)
print(f"Best move: {best_move} (Score: {score})")
```
