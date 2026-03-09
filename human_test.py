import curses
import time

from tetris import TetrisEnv


def draw_board(stdscr, env, rot_idx, col):
    stdscr.clear()
    rows, cols = env.rows, env.cols
    board = env.get_board()

    # compute rotated coords for preview
    base = env.PIECES[env.current_piece]
    rotations = env._rotations(base)
    rot = rotations[rot_idx % len(rotations)]

    # clamp col
    max_x = max(x for x, y in rot)
    width = max_x + 1
    col = max(0, min(col, cols - width))

    # compute drop row for preview
    row = 0
    while not env._would_collide(rot, row + 1, col):
        row += 1

    # render header
    stdscr.addstr(0, 0, f'Piece: {env.current_piece}  Rotation: {rot_idx % len(rotations)}  Col: {col}')
    stdscr.addstr(1, 0, 'Controls: ←/→ move  ↑ rotate  ↓ rotate back  Space place  q quit')

    # draw board with preview
    for r in range(rows):
        line = ''
        for c in range(cols):
            ch = '.' if board[r][c] == 0 else '#'
            line += ch
        stdscr.addstr(3 + r, 0, line)

    # overlay preview using 'O' char
    for x, y in rot:
        rr = row + y
        cc = col + x
        if 0 <= rr < rows and 0 <= cc < cols:
            stdscr.addstr(3 + rr, cc, 'O')

    stdscr.refresh()


def human_play(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.keypad(True)

    env = TetrisEnv(rows=20, cols=10)
    env.spawn_piece()

    rot_idx = 0
    col = 0

    while True:
        draw_board(stdscr, env, rot_idx, col)
        key = stdscr.getch()
        if key == ord('q'):
            break
        elif key in (curses.KEY_LEFT, ord('a')):
            col = max(0, col - 1)
        elif key in (curses.KEY_RIGHT, ord('d')):
            col = min(env.cols - 1, col + 1)
        elif key in (curses.KEY_UP, ord('w')):
            rot_idx = (rot_idx + 1) % 4
        elif key in (curses.KEY_DOWN, ord('s')):
            rot_idx = (rot_idx - 1) % 4
        elif key in (ord(' '), ord('\n')):
            # place current piece with chosen rotation and column
            # clamp column relative to rotation width
            base = env.PIECES[env.current_piece]
            rotations = env._rotations(base)
            rot = rotations[rot_idx % len(rotations)]
            max_x = max(x for x, y in rot)
            width = max_x + 1
            col = max(0, min(col, env.cols - width))

            # convert to single integer action index: index = col * 4 + rotation
            action_index = col * 4 + (rot_idx % 4)
            state, reward, terminated, truncated, info = env.step(action_index)
            if terminated:
                draw_board(stdscr, env, rot_idx, col)
                stdscr.addstr(3 + env.rows + 1, 0, 'Game Over. Press any key to exit.')
                stdscr.getch()
                break
            # reset controls for next piece
            rot_idx = 0
            col = 0


def main():
    try:
        curses.wrapper(human_play)
    except Exception as e:
        print('Failed to start curses UI:', e)
        print('Make sure you run this script in a terminal that supports curses.')


if __name__ == '__main__':
    main()
