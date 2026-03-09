from tetris import TetrisEnv


def demo_random_play(steps=5):
    """Demo loop: env doesn't auto-supply pieces — we sample pieces/actions
    with `sample_random_piece_action` and apply them. State = board + piece.
    """
    env = TetrisEnv(rows=10, cols=10)
    for step in range(steps):
        piece = env.current_piece
        action = env.sample_random_action(allow_game_over=False)
        if action is None:
            print('No valid placements for current piece; stopping.')
            break
        rot, col = action
        print(f'Step {step+1}: piece={piece} rotation={rot} col={col}')
        state, reward, terminated, truncated, info = env.step(rot, col)
        if terminated:
            print('Placement collided immediately (game over).')
            break
        # print state: board + next piece
        board, next_piece = state
        print('Board state after placement:')
        env.pretty_print()
        print('Next piece:', next_piece, 'reward:', reward)


if __name__ == '__main__':
    demo_random_play(steps=8)
