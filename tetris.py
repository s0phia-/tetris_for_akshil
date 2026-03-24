import os
from copy import deepcopy
import networkx as nx
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TetrisEnv(gym.Env):
    """Minimal Tetris environment 

    Observation: dict with 'board' (MultiBinary rows×cols) and 'piece' (Discrete)
    Action:      integer index mapping to (rotation, col) via action_index_to_pair()
    Reward:      number of lines cleared on each step
    Termination: piece collides immediately on placement (board overflow)
    """

    metadata = {"render_modes": ["ansi"]}

    PIECES = {
        'I': [(0,1),(1,1),(2,1),(3,1)],
        'O': [(0,0),(1,0),(0,1),(1,1)],
        'T': [(0,0),(1,0),(2,0),(1,1)],
        'S': [(1,0),(2,0),(0,1),(1,1)],
        'Z': [(0,0),(1,0),(1,1),(2,1)],
        'J': [(0,0),(0,1),(1,1),(2,1)],
        'L': [(2,0),(0,1),(1,1),(2,1)],
    }

    def __init__(
        self,
        rows: int = 20,
        cols: int = 10,
        piece_types: list | int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.render_mode = render_mode
        self.spec = None

        all_pieces = list(self.PIECES.keys())
        if piece_types is None:
            self.available_pieces = all_pieces
        elif isinstance(piece_types, int):
            self.available_pieces = all_pieces[:piece_types]
        else:
            provided = list(piece_types)
            self.available_pieces = provided

        self.piece_to_index = {p: i for i, p in enumerate(self.available_pieces)}
        self.index_to_piece = {i: p for p, i in self.piece_to_index.items()}

        self.observation_space = spaces.Dict({
            "board": spaces.MultiBinary((self.rows, self.cols)),
            "piece": spaces.Discrete(len(self.available_pieces)),
        })
        self.action_space = spaces.Discrete(4 * self.cols)

        # internal state — uninitialised until reset() is called
        self._grid: list[list[int]] = []
        self._current_piece: str = ""
        self.np_random, _ = gym.utils.seeding.np_random()

    # ------------------------------------------------------------------
    # Gymnasium core API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._grid = [[0] * self.cols for _ in range(self.rows)]
        self._current_piece = self.np_random.choice(self.available_pieces)
        return self._obs(), {}

    def step(self, action: int):
        rotation, col = self.action_index_to_pair(int(action))
        rot = self._get_rotation(self._current_piece, rotation)
        width = max(x for x, y in rot) + 1

        if not (0 <= col <= self.cols - width):
            return self._obs(), 0, False, False, {"invalid_action": True}

        if self._would_collide_grid(rot, 0, col):
            return self._obs(), 0, True, False, {"invalid_action": False}

        row = 0
        while not self._would_collide_grid(rot, row + 1, col):
            row += 1

        for x, y in rot:
            self._grid[row + y][col + x] = 1

        reward = self._clear_full_lines()
        self._current_piece = self.np_random.choice(self.available_pieces)
        return self._obs(), reward, False, False, {"invalid_action": False}

    def render(self):
        if self.render_mode == "ansi":
            return "\n".join(
                "".join("#" if c else "." for c in row) for row in self._grid
            )

    def close(self):
        pass

    @property
    def unwrapped(self):
        return self

    def __repr__(self):
        return (
            f"TetrisEnv(rows={self.rows}, cols={self.cols}, "
            f"pieces={self.available_pieces}, current={self._current_piece!r})"
        )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _obs(self):
        return {
            "board": np.array(self._grid, dtype=np.int8),
            "piece": self.piece_to_index[self._current_piece],
        }

    # ------------------------------------------------------------------
    # Piece geometry
    # ------------------------------------------------------------------

    def _rotations(self, coords: list) -> list:
        """Return all unique normalised rotation states for a piece."""
        states = []
        current = coords[:]
        for _ in range(4):
            min_x = min(x for x, y in current)
            min_y = min(y for x, y in current)
            norm = sorted((x - min_x, y - min_y) for x, y in current)
            if norm not in states:
                states.append(norm)
            current = [(y, -x) for x, y in current]
        return states

    def _get_rotation(self, piece: str, rotation: int) -> list:
        rots = self._rotations(self.PIECES[piece])
        return rots[rotation % len(rots)]

    def _get_all_actions_for_piece(self, piece: str) -> list:
        actions = []
        for ri, rot in enumerate(self._rotations(self.PIECES[piece])):
            width = max(x for x, y in rot) + 1
            for col in range(self.cols - width + 1):
                actions.append((ri, col))
        return actions

    def action_index_to_pair(self, index: int) -> tuple[int, int]:
        """Map integer action to (rotation, col): rotation = index % 4, col = index // 4."""
        return (index % 4, index // 4)

    # ------------------------------------------------------------------
    # Board helpers — two variants: live grid and arbitrary board
    # ------------------------------------------------------------------

    def _would_collide_grid(self, rot: list, row: int, col: int) -> bool:
        for x, y in rot:
            r, c = row + y, col + x
            if r >= self.rows or self._grid[r][c] == 1:
                return True
        return False

    def _would_collide_board(self, board: list, rot: list, row: int, col: int) -> bool:
        for x, y in rot:
            r, c = row + y, col + x
            if r >= self.rows or board[r][c] == 1:
                return True
        return False

    def _clear_full_lines(self) -> int:
        clean = [row for row in self._grid if not all(row)]
        removed = self.rows - len(clean)
        self._grid = [[0] * self.cols] * removed + clean
        return removed

    def _simulate_place(self, board, piece, rotation, col):
        """Simulate placing piece on an arbitrary board; does not mutate self._grid.

        Returns (new_board, collided, lines_cleared, placed_rows).
        """
        rot = self._get_rotation(piece, rotation)
        width = max(x for x, y in rot) + 1

        if not (0 <= col <= self.cols - width):
            return deepcopy(board), True, 0, []

        if self._would_collide_board(board, rot, 0, col):
            return deepcopy(board), True, 0, []

        row = 0
        while not self._would_collide_board(board, rot, row + 1, col):
            row += 1

        new_board = [list(r) for r in deepcopy(board)]
        placed_rows = []
        for x, y in rot:
            r, c = row + y, col + x
            new_board[r][c] = 1
            placed_rows.append(r)

        clean = [r for r in new_board if not all(r)]
        removed = self.rows - len(clean)
        new_board = [[0] * self.cols] * removed + clean
        return new_board, False, removed, placed_rows

    # ------------------------------------------------------------------
    # State transition graph (offline planning utility)
    # ------------------------------------------------------------------

    def _hash_state(self, state) -> tuple:
        board, piece = state
        return (tuple(tuple(row) for row in board), piece or "<TERMINAL>")

    def get_successor_states_given_action(self, state, action):
        """Return list of (next_state, prob) for a (board, piece) state."""
        board, piece = state
        if piece is None:
            return []

        rotation, col = action
        new_board, collided, _, _ = self._simulate_place(board, piece, rotation, col)

        if collided:
            return [((deepcopy(board), None), 1.0)]

        p = 1.0 / len(self.available_pieces)
        return [((deepcopy(new_board), npiece), p) for npiece in self.available_pieces]

    def generate_stg(self, directed: bool = True, save_path: str | None = None):
        """Generate the state-transition graph. Node feature: current tetrimino only."""

        seen = set()
        all_states = []
        empty_board = [[0] * self.cols for _ in range(self.rows)]
        current_states = [(deepcopy(empty_board), p) for p in self.available_pieces]

        while current_states:
            next_states = []
            for state in current_states:
                h = self._hash_state(state)
                if h in seen:
                    continue
                seen.add(h)
                all_states.append(state)
                _, piece = state
                if piece is None:
                    continue
                for action in self._get_all_actions_for_piece(piece):
                    for succ_state, _ in self.get_successor_states_given_action(state, action):
                        next_states.append(succ_state)
            current_states = deepcopy(next_states)

        stg = nx.DiGraph() if directed else nx.Graph()

        for state in all_states:
            _, piece = state
            stg.add_node(self._hash_state(state), piece=str(piece), is_terminal=int(piece is None))

        for state in all_states:
            h = self._hash_state(state)
            _, piece = state
            if piece is None:
                continue
            for action in self._get_all_actions_for_piece(piece):
                for succ_state, _ in self.get_successor_states_given_action(state, action):
                    hs = self._hash_state(succ_state)
                    if not stg.has_edge(h, hs):
                        stg.add_edge(h, hs)

        if save_path is not None:
            dir_name = os.path.dirname(save_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            nx.write_gexf(stg, save_path)

        return stg
    