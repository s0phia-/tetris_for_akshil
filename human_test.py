import curses
from tetris import TetrisEnv


def draw_board(stdscr, board, piece, rot, col, drop_row, rot_idx, num_rots):
    stdscr.clear()
    rows = len(board)
    cols = len(board[0])

    stdscr.addstr(0, 0, f"Piece: {piece}  Rotation: {rot_idx % num_rots}  Col: {col}")
    stdscr.addstr(1, 0, "Controls: ←/→ move  ↑ rotate  ↓ rotate back  Space place  q quit")

    preview = {(col + x, drop_row + y) for x, y in rot}

    for r in range(rows):
        line = ""
        for c in range(cols):
            if (c, r) in preview:
                line += "O"
            elif board[r][c]:
                line += "#"
            else:
                line += "."
        stdscr.addstr(3 + r, 0, line)

    stdscr.refresh()


def human_play(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.keypad(True)

    env = TetrisEnv(rows=4, cols=4)
    obs, _ = env.reset()
    rot_idx = 0
    col = 0

    while True:
        piece = env.index_to_piece[obs["piece"]]
        board = obs["board"].tolist()
        rots = env._rotations(env.PIECES[piece])
        rot = rots[rot_idx % len(rots)]
        width = max(x for x, y in rot) + 1
        col = max(0, min(col, env.cols - width))

        # compute ghost drop row
        drop_row = 0
        while not env._would_collide_board(board, rot, drop_row + 1, col):
            drop_row += 1

        draw_board(stdscr, board, piece, rot, col, drop_row, rot_idx, len(rots))

        key = stdscr.getch()

        if key == ord("q"):
            break
        elif key in (curses.KEY_LEFT, ord("a")):
            col = max(0, col - 1)
        elif key in (curses.KEY_RIGHT, ord("d")):
            col = min(env.cols - 1, col + 1)
        elif key in (curses.KEY_UP, ord("w")):
            rot_idx = (rot_idx + 1) % 4
        elif key in (curses.KEY_DOWN, ord("s")):
            rot_idx = (rot_idx - 1) % 4
        elif key in (ord(" "), ord("\n")):
            action = col * 4 + (rot_idx % 4)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                stdscr.addstr(3 + env.rows + 1, 0, "Game Over. Press any key to exit.")
                stdscr.getch()
                break
            rot_idx = 0
            col = 0


if __name__ == "__main__":
    curses.wrapper(human_play)