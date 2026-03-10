#!/usr/bin/env python3
"""Generic STG generator for TetrisEnv with top-level parameters.

Edit the parameters below to configure the board and pieces. No CLI required.

Potential tetriminos (letters you can use in `PIECES`):
  I - straight 4-block line
  O - 2x2 square
  T - T-shape
  S - skew (S)
  Z - skew (Z)
  J - J-shape (like L mirrored)
  L - L-shape

Examples:
  - Use first 3 pieces: PIECES = '3'
  - Use specific pieces: PIECES = 'I,O,T'

Outputs: {OUT}.gexf and (if matplotlib installed) {OUT}.png


Features
    Features:
    0: rows_with_holes
    1: column_transitions
    2: holes
    3: landing height
    4: cumulative_wells
    5: row_transitions
    6: eroded pieces
    7: hole_depth
    
"""
import os

# ----------------- USER CONFIG -----------------
# Board size
ROWS = 4
COLS = 4

# Pieces: either an int N (use first N pieces ['I','O','T','S','Z','J','L'])
# or a comma-separated list of piece keys, e.g. 'I,O,T'
# Examples: PIECES = '3'  or PIECES = 'I,O,T'
PIECES = '2'

# Output prefix
OUT = f'stg_{ROWS}x{COLS}_{PIECES}'

# Do not attempt to save PNG visualization
NO_PNG = False
# ------------------------------------------------

from tetris import TetrisEnv


def parse_pieces_arg(arg: str):
    if arg is None:
        return None
    arg = str(arg).strip()
    if arg.isdigit():
        return int(arg)
    parts = [p.strip().upper() for p in arg.split(',') if p.strip()]
    return parts


def main():
    pieces_arg = parse_pieces_arg(PIECES)

    try:
        import networkx as nx
    except Exception:
        print('networkx is required; install with: pip install networkx')
        return

    try:
        import matplotlib.pyplot as plt
        has_plt = True
    except Exception:
        has_plt = False

    piece_types = pieces_arg
    if isinstance(piece_types, list):
        piece_types = [p.upper() for p in piece_types]

    env = TetrisEnv(rows=ROWS, cols=COLS, piece_types=piece_types)
    print('Using available pieces:', env.available_pieces)
    print(f'Generating STG for {ROWS}x{COLS}...')

    stg = env.generate_stg(directed=True)

    out_gexf = OUT + '.gexf'
    nx.write_gexf(stg, out_gexf)
    print(f'Saved graph to {out_gexf} (|V|={stg.number_of_nodes()}, |E|={stg.number_of_edges()})')

    if not NO_PNG and has_plt:
        print('Creating PNG visualization (may be slow)...')
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(stg, seed=1)
            nx.draw(stg, pos, node_size=20, linewidths=0.1)
            out_png = OUT + '.png'
            plt.savefig(out_png, dpi=150)
            print(f'Saved visualization to {out_png}')
        except Exception as e:
            print('Failed to draw graph:', e)


if __name__ == '__main__':
    main()
