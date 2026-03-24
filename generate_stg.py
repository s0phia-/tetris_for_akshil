"""Generate a Tetris STG and save as .gexf (and optionally .png).

Pieces: 'I','O','T','S','Z','J','L'
PIECES = 3       → first 3 pieces
"""
from tetris import TetrisEnv
import matplotlib.pyplot as plt


# --- config ---
ROWS   = 4
COLS   = 4
PIECES = 3
OUT    = f"stg_{ROWS}x{COLS}_{PIECES}"
NO_PNG = False
# --------------

def parse_pieces(arg) -> int | list | None:
    if arg is None:
        return None
    if isinstance(arg, int):
        return arg
    arg = str(arg).strip()
    return int(arg) if arg.isdigit() else [p.strip().upper() for p in arg.split(",") if p.strip()]

def main():
    env = TetrisEnv(rows=ROWS, cols=COLS, piece_types=parse_pieces(PIECES))
    print(f"Generating STG for {ROWS}x{COLS}...")

    stg = env.generate_stg(directed=True, save_path=f"{OUT}.gexf")
    print(f"Saved {OUT}.gexf  (|V|={stg.number_of_nodes()}, |E|={stg.number_of_edges()})")

    if not NO_PNG:
        import networkx as nx
        plt.figure(figsize=(12, 8))
        nx.draw(stg, nx.spring_layout(stg, seed=1), node_size=20, linewidths=0.1)
        plt.savefig(f"{OUT}.png", dpi=150)
        print(f"Saved {OUT}.png")

if __name__ == "__main__":
    main()