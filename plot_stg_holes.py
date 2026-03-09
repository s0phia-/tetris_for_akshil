#!/usr/bin/env python3
"""Generate STG and plot nodes colored by number of holes (feature index 2).

Edit the parameters below if you want a different board or pieces.
"""
ROWS = 3
COLS = 4
PIECES = '2'  # first 3 pieces I,O,T
OUT_PNG = 'stg_landing_height.png'

from tetris import TetrisEnv
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import deepcopy


def main():
    piece_types = PIECES
    if isinstance(piece_types, str) and piece_types.isdigit():
        piece_types = int(piece_types)

    env = TetrisEnv(rows=ROWS, cols=COLS, piece_types=piece_types)
    print('Generating STG...')
    stg = env.generate_stg(directed=True)
    print(f'Nodes: {stg.number_of_nodes()}, Edges: {stg.number_of_edges()}')

    # Extract landing-height feature (feature_3) for each node
    nodes = list(stg.nodes())
    values = []
    for n in nodes:
        data = stg.nodes[n]
        v = data.get('feature_3', 0.0)
        values.append(float(v))

    # Debug: print hole statistics and dump per-node values
    from collections import Counter
    counter = Counter(values)
    print('Landing-height values: min=', min(values) if values else None, 'max=', max(values) if values else None)
    print('Unique landing-height values count=', len(counter))
    print('Value counts sample:', dict(list(counter.items())[:10]))
    try:
        with open('node_landing_height.csv', 'w') as fh:
            fh.write('node,landing_height\n')
            for n, v in zip(nodes, values):
                fh.write(f'"{n}",{v}\n')
        print('Wrote node_landing_height.csv')
    except Exception as e:
        print('Failed to write CSV:', e)

    # layout
    pos = nx.spring_layout(stg, seed=1)

    plt.figure(figsize=(12, 9))
    ax = plt.gca()

    # draw edges
    nx.draw_networkx_edges(stg, pos, alpha=0.3, width=0.5, ax=ax)

    # draw nodes with colormap
    cmap = plt.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=min(values) if values else 0, vmax=max(values) if values else 1)
    node_colors = [cmap(norm(v)) for v in values]
    nx.draw_networkx_nodes(stg, pos, node_color=node_colors, node_size=50, ax=ax)

    # colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(values)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Landing height (feature_3)')

    plt.title(f'STG nodes colored by landing height (rows={ROWS}, cols={COLS}, pieces={env.available_pieces})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print('Saved', OUT_PNG)


if __name__ == '__main__':
    main()
