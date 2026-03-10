import random
from copy import deepcopy
import numpy as np
import os


class TetrisEnv:
    """Minimal Tetris environment.

    Grid values: 0 = empty, 1 = filled

    Pieces are specified by a letter: 'I','O','T','S','Z','J','L'.
    Actions: place a specified piece with a rotation index and a column index
    (the column is the leftmost column of the piece's bounding box).

    Pieces are placed instantly (they do not fall each timestep). The piece
    is dropped until it rests (settles) and then locked in place; completed
    lines are cleared.
    """

    PIECES = {
        'I': [(0,1),(1,1),(2,1),(3,1)],
        'O': [(0,0),(1,0),(0,1),(1,1)],
        'T': [(0,0),(1,0),(2,0),(1,1)],
        'S': [(1,0),(2,0),(0,1),(1,1)],
        'Z': [(0,0),(1,0),(1,1),(2,1)],
        'J': [(0,0),(0,1),(1,1),(2,1)],
        'L': [(2,0),(0,1),(1,1),(2,1)],
    }

    def __init__(self, rows=20, cols=10, piece_types: list | int | None = None):
        """Create a TetrisEnv.

        Args:
            rows, cols: board size
            piece_types: if None, use all pieces. If int n, use the first n piece types.
                         If list, use that list of piece keys (subset of 'I','O','T','S','Z','J','L').
        """
        self.rows = rows
        self.cols = cols
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        # determine available pieces
        all_pieces = list(self.PIECES.keys())
        if piece_types is None:
            self.available_pieces = all_pieces
        elif isinstance(piece_types, int):
            if piece_types <= 0 or piece_types > len(all_pieces):
                raise ValueError('num piece_types must be between 1 and %d' % len(all_pieces))
            self.available_pieces = all_pieces[:piece_types]
        else:
            # assume iterable/list of piece keys
            provided = list(piece_types)
            for p in provided:
                if p not in all_pieces:
                    raise ValueError('Unknown piece in piece_types: %r' % p)
            self.available_pieces = provided

        # current piece supplied by the environment
        self.current_piece = random.choice(self.available_pieces)
        self.last_cleared_lines = 0

        # piece -> index mapping for compact observations
        self.piece_to_index = {p: i for i, p in enumerate(self.available_pieces)}
        self.index_to_piece = {i: p for p, i in self.piece_to_index.items()}

        # expose Gym-style action_space and observation_space when possible
        try:
            from gymnasium.spaces import Discrete, MultiBinary, Dict as SpaceDict
            self.action_space = Discrete(4 * self.cols)
            # observation: board as MultiBinary(rows,cols) and piece as Discrete(num_pieces)
            self.observation_space = SpaceDict({
                'board': MultiBinary((self.rows, self.cols)),
                'piece': Discrete(len(self.available_pieces)),
            })
        except Exception:
            # fallback: simple descriptive dicts
            self.action_space = {'type': 'discrete', 'n': 4 * self.cols, 'mapping': 'rotation=index%4, col=index//4'}
            self.observation_space = {
                'board': {'shape': (self.rows, self.cols), 'dtype': 'int', 'values': [0, 1]},
                'piece': {'choices': list(self.available_pieces)}
            }

    def reset(self):
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

    def _rotations(self, coords):
        # generate up to 4 unique rotation states (normalized)
        states = []
        current = coords[:]
        for _ in range(4):
            # normalize so min x/y are 0
            min_x = min(x for x, y in current)
            min_y = min(y for x, y in current)
            norm = sorted(((x - min_x, y - min_y) for x, y in current))
            if norm not in states:
                states.append(norm)
            # rotate 90deg clockwise: (x,y) -> (y, -x)
            current = [(y, -x) for x, y in current]
        return states

    def valid_col(self, piece_coords, col):
        max_x = max(x for x, y in piece_coords)
        min_x = min(x for x, y in piece_coords)
        width = max_x - min_x + 1
        return 0 <= col and (col + width) <= self.cols

    def _would_collide(self, piece_coords, row, col):
        for x, y in piece_coords:
            r = row + y
            c = col + x
            if r >= self.rows:
                return True
            if self.grid[r][c] == 1:
                return True
        return False

    def place_piece(self, piece, rotation, col):
        """Place `piece` with rotation index `rotation` aligned so its
        bounding-box left is at column `col`.

        Returns:
          - True if placed successfully
          - False if placement causes immediate overflow (game over)
        """
        if piece not in self.PIECES:
            raise ValueError('Unknown piece: %r' % piece)

        base = self.PIECES[piece]
        rotations = self._rotations(base)
        rot = rotations[rotation % len(rotations)]

        # rot coordinates are normalized so min x/y == 0. width/height:
        max_x = max(x for x, y in rot)
        max_y = max(y for x, y in rot)
        width = max_x + 1

        if not (0 <= col <= self.cols - width):
            raise ValueError('Invalid column for this rotation: %d' % col)

        # drop instantly: start at row 0 and move down until collision
        row = 0
        # if collides already at row 0 -> game over (cannot place)
        if self._would_collide(rot, row, col):
            return False

        while True:
            if self._would_collide(rot, row + 1, col):
                break
            row += 1

        # lock piece at final row
        for x, y in rot:
            r = row + y
            c = col + x
            self.grid[r][c] = 1

        # clear full lines and record how many were cleared
        removed = self._clear_full_lines()
        self.last_cleared_lines = removed
        return True

    def _clear_full_lines(self):
        new_grid = [row for row in self.grid if not all(cell == 1 for cell in row)]
        removed = self.rows - len(new_grid)
        for _ in range(removed):
            new_grid.insert(0, [0] * self.cols)
        self.grid = new_grid
        return removed

    def _hash_state(self, state):
        """Make a hashable representation of state (board, piece)."""
        board, piece = state
        # board might be list of lists; convert to tuple of tuples
        board_t = tuple(tuple(int(x) for x in row) for row in board)
        piece_t = piece if piece is not None else "<TERMINAL>"
        return (board_t, piece_t)

    def _simulate_place(self, board, piece, rotation, col):
        """Simulate placing `piece` with `rotation` at `col` on given board.

        Returns (new_board, collided, lines_cleared).
        """
        rows = self.rows
        cols = self.cols
        base = self.PIECES[piece]
        rotations = self._rotations(base)
        rot = rotations[rotation % len(rotations)]

        max_x = max(x for x, y in rot)
        width = max_x + 1
        if not (0 <= col <= cols - width):
            # invalid column -> treat as collision/invalid placement
            return deepcopy(board), True, 0

        def would_collide_at(bd, row):
            for x, y in rot:
                r = row + y
                c = col + x
                if r >= rows:
                    return True
                if bd[r][c] == 1:
                    return True
            return False

        # check immediate collision
        if would_collide_at(board, 0):
            return deepcopy(board), True, 0, []

        row = 0
        while not would_collide_at(board, row + 1):
            row += 1

        new_board = [list(r) for r in deepcopy(board)]
        placed_rows = []
        for x, y in rot:
            r = row + y
            c = col + x
            new_board[r][c] = 1
            placed_rows.append(r)

        # clear full lines
        new_grid = [row for row in new_board if not all(cell == 1 for cell in row)]
        removed = rows - len(new_grid)
        for _ in range(removed):
            new_grid.insert(0, [0] * cols)

        return new_grid, False, removed, placed_rows

    def get_successor_states_given_action(self, state, action):
        """Given a state (board, piece) and an action (rotation, col),
        return a list of (next_state, prob) pairs.
        - If the state is terminal (piece is None), return []
        - If placement immediately collides -> terminal successor (piece=None)
        - Otherwise, successor is (new_board, next_piece) for each possible next_piece,
          each with equal probability (1 / number_of_piece_types).
        """
        board, piece = state
        if piece is None:
            return []

        rotation, col = action
        # simulate
        new_board, collided, _, _ = self._simulate_place(board, piece, rotation, col)
        if collided:
            # terminal state: piece set to None
            return [((deepcopy(board), None), 1.0)]

        # spawn next piece uniformly from available pieces
        successors = []
        pieces = list(self.available_pieces)
        p = 1.0 / len(pieces)
        for npiece in pieces:
            successors.append(((deepcopy(new_board), npiece), p))
        return successors

    def _get_all_actions_for_piece(self, piece):
        """Return list of (rotation_index, column) valid action pairs for `piece`."""
        base = self.PIECES[piece]
        rotations = self._rotations(base)
        actions = []
        for ri, rot in enumerate(rotations):
            max_x = max(x for x, y in rot)
            width = max_x + 1
            for col in range(0, self.cols - width + 1):
                actions.append((ri, col))
        return actions

    def calc_lowest_free_rows(self, representation):
        """Return lowest filled-row index+1 per column (0 if column empty).

        `representation` may be a list-of-rows or a NumPy 2D array.
        """
        num_rows = len(representation)
        n_cols = len(representation[0]) if num_rows else 0
        lowest_free_rows = [0] * n_cols
        for col_ix in range(n_cols):
            lowest = 0
            for row_ix in range(num_rows - 1, -1, -1):
                if representation[row_ix][col_ix]:
                    lowest = row_ix + 1
                    break
            lowest_free_rows[col_ix] = lowest
        return lowest_free_rows

    def get_feature_values_jitted(self, lowest_free_rows, representation, num_rows, num_columns):
        """
        Compute feature subset: rows_with_holes, column_transitions, holes, cumulative_wells, row_transitions, hole_depth
        Returns list in that order.
        """
        board = np.array(representation, dtype=int)
        rows, cols = board.shape

        # Rows with holes: number of rows with at least one hole
        rows_with_holes = 0
        for r in range(rows):
            has_hole = False
            for c in range(cols):
                if board[r, c] == 0 and np.any(board[:r, c] == 1):
                    has_hole = True
                    break
            if has_hole:
                rows_with_holes += 1

        # Column transitions
        column_transitions = 0
        for c in range(cols):
            prev = 1  # Assume outside bottom is full
            for r in range(rows):
                curr = board[r, c]
                if curr != prev:
                    column_transitions += 1
                prev = curr
            if prev == 0:
                column_transitions += 1  # Outside top is empty

        # Holes and hole depth
        holes = 0
        hole_depth = 0
        for c in range(cols):
            for r in range(rows):
                if board[r, c] == 0 and np.any(board[:r, c] == 1):
                    holes += 1
                    # Count filled cells directly above
                    above = board[:r, c][::-1]
                    for cell in above:
                        if cell == 1:
                            hole_depth += 1
                        else:
                            break

        # Cumulative wells
        cumulative_wells = 0
        for c in range(cols):
            well_depth = 0
            for r in range(rows):
                left = 1 if c == 0 else board[r, c - 1]
                right = 1 if c == cols - 1 else board[r, c + 1]
                if board[r, c] == 0 and left == 1 and right == 1:
                    well_depth += 1
                    cumulative_wells += well_depth
                else:
                    well_depth = 0

        # Row transitions
        row_transitions = 0
        for r in range(rows):
            prev = 1  # Assume outside left is full
            for c in range(cols):
                curr = board[r, c]
                if curr != prev:
                    row_transitions += 1
                prev = curr
            if prev == 0:
                row_transitions += 1  # Outside right is full

        return [rows_with_holes, column_transitions, holes, cumulative_wells, row_transitions, hole_depth]

    def compute_features_from_board(self, board, landing_height_override=None):
        """Compute 8 features for given board.

        Returns list of 8 feature values in this order:
        0: rows_with_holes
        1: column_transitions
        2: holes
        3: landing_height (approx: mean column height)
        4: cumulative_wells
        5: row_transitions
        6: eroded_pieces (approx: 0)
        7: hole_depth
        """
        # board is list of rows; convert to numeric
        num_rows = len(board)
        num_cols = len(board[0]) if num_rows > 0 else 0
        lowest_free_rows = self.calc_lowest_free_rows(board)
        vals = self.get_feature_values_jitted(lowest_free_rows, board, num_rows, num_cols)
        # vals order: rows_with_holes, column_transitions, holes, cumulative_wells, row_transitions, hole_depth
        rows_with_holes, column_transitions, holes, cumulative_wells, row_transitions, hole_depth = vals
        # landing height: allow explicit override (from placement), otherwise prefer explicit attributes
        if landing_height_override is not None:
            landing_height = float(landing_height_override)
        elif hasattr(self, 'anchor_row') and hasattr(self, 'landing_height_bonus') and self.anchor_row is not None and self.landing_height_bonus is not None:
            landing_height = float(self.anchor_row + self.landing_height_bonus + 1)
        else:
            # fallback: use mean height of non-empty columns (better proxy than including zeros)
            nonzero = [h for h in lowest_free_rows if h > 0]
            if nonzero:
                landing_height = float(sum(nonzero) / len(nonzero))
            else:
                landing_height = 0.0
        eroded_pieces = 0.0
        features = [float(rows_with_holes), float(column_transitions), float(holes), float(landing_height), float(cumulative_wells), float(row_transitions), float(eroded_pieces), float(hole_depth)]
        return features

    def generate_stg(self, directed: bool = True, save_path: str | None = None):
        """Generate the state-transition graph for the Tetris environment.

        Nodes are hashed states (board, piece). Terminal states have piece=None.
        """
        seen = set()
        all_states = []
        current_states = []

        # initial states: empty board with every possible starting piece (respect available_pieces)
        empty_board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for piece in self.available_pieces:
            current_states.append((deepcopy(empty_board), piece))

        while current_states:
            next_states = []
            for state in current_states:
                h = self._hash_state(state)
                if h in seen:
                    seen.add(h)
                    all_states.append(state)
                    continue
                board, piece = state
                if piece is None:
                    continue
                features = self.compute_features_from_board(board)
                actions = self._get_all_actions_for_piece(piece)
                for action in actions:
                    successors = self.get_successor_states_given_action(state, action)
                    for succ_state, _ in successors:
                        next_states.append(succ_state)
                    next_states.append(succ_state)
            current_states = deepcopy(next_states)

        # build graph (import networkx lazily to avoid hard dependency at module import)
        import networkx as nx

        stg = nx.DiGraph() if directed else nx.Graph()
        # create nodes with baseline features (landing height may be updated below)
        node_landing_heights = {}
        for state in all_states:
            h = self._hash_state(state)
            board, piece = state
            is_terminal = int(piece is None)
            # compute baseline features for this board
            try:
                features = self.compute_features_from_board(board)
            except Exception:
                features = [0] * 8
            attrs = {f'feature_{i}': float(features[i]) for i in range(len(features))}
            attrs.update({'piece': str(piece), 'is_terminal': is_terminal})
            stg.add_node(h, **attrs)

        # build edges and record landing heights for successor nodes where possible
        for state in all_states:
            h = self._hash_state(state)
            board, piece = state
            if piece is None:
                continue
            actions = self._get_all_actions_for_piece(piece)
            for action in actions:
                # simulate to get placed_rows for computing landing height
                rotation, col = action
                sim_board, collided, _, placed_rows = self._simulate_place(board, piece, rotation, col)
                successors = self.get_successor_states_given_action(state, action)
                for succ_state, _ in successors:
                    hs = self._hash_state(succ_state)
                    if not stg.has_edge(h, hs):
                        stg.add_edge(h, hs)
                    # if placement didn't collide, compute landing height and accumulate
                    if not collided and placed_rows:
                        # convert row indices to heights from bottom (0 = bottom)
                        heights = [self.rows - r - 1 for r in placed_rows]
                        y1 = float(min(heights))
                        y2 = float(max(heights))
                        lh = (y1 + y2) / 2.0
                        node_landing_heights.setdefault(hs, []).append(lh)

        # average landing heights from predecessors and update node attributes
        for node_hash, llist in node_landing_heights.items():
            avg_lh = float(sum(llist) / len(llist))
            if node_hash in stg.nodes:
                stg.nodes[node_hash][f'feature_3'] = avg_lh

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            nx.write_gexf(stg, save_path)
        return stg

    def get_state(self):
        """Return the environment state as a tuple `(board, piece)` where:
        - `board` is a deep-copy 2D list of 0/1 values
        - `piece` is the current tetrimino supplied by the environment
        """
        return ([row[:] for row in self.grid], self.current_piece)

    def get_board(self):
        """Return just the board (deep-copy) for compatibility."""
        return [row[:] for row in self.grid]

    def spawn_piece(self, piece=None):
        """Set `self.current_piece`. If `piece` is None choose uniformly.
        Returns the spawned piece."""
        if piece is None:
            piece = random.choice(self.available_pieces)
        if piece not in self.available_pieces:
            raise ValueError('Unknown piece: %r' % piece)
        self.current_piece = piece
        return piece

    def sample_random_action(self, allow_game_over=False):
        """Return a random valid (rotation_index, column) for the current piece,
        or `None` if no valid placement exists."""
        piece = self.current_piece
        base = self.PIECES[piece]
        rotations = self._rotations(base)
        valid_actions = []

        for ri, rot in enumerate(rotations):
            max_x = max(x for x, y in rot)
            width = max_x + 1
            for col in range(0, self.cols - width + 1):
                if not self.valid_col(rot, col):
                    continue
                # if immediate collision at row 0
                if self._would_collide(rot, 0, col):
                    if allow_game_over:
                        valid_actions.append((ri, col))
                    continue

                # simulate drop
                row = 0
                while not self._would_collide(rot, row + 1, col):
                    row += 1
                valid_actions.append((ri, col))

        if not valid_actions:
            return None
        return random.choice(valid_actions)

    def sample_random_piece_action(self, allow_game_over=False):
        """Deprecated: kept for backward compatibility. Use `spawn_piece` then
        `sample_random_action` (which samples for the current piece).
        This method samples a piece and returns (piece, rotation, col) or None."""
        pieces = list(self.available_pieces)
        random.shuffle(pieces)
        for p in pieces:
            # temporarily test for piece p
            saved = self.current_piece
            try:
                self.spawn_piece(p)
                action = self.sample_random_action(allow_game_over=allow_game_over)
            finally:
                self.current_piece = saved
            if action is not None:
                ri, col = action
                return (p, ri, col)
        return None

    def get_action_space(self):
        """Return a fixed action mapping as a list of (rotation, column) pairs.

        The environment exposes a single integer action space where each
        index corresponds to a `(rotation, column)` pair. Use
        `action_index_to_pair(idx)` to convert indices.
        """
        return [(r, c) for r in range(4) for c in range(self.cols)]

    def action_count(self):
        """Return the size of the fixed integer action space."""
        return 4 * self.cols

    def action_index_to_pair(self, index: int):
        """Map an integer action `index` to a `(rotation, col)` pair.

        Indices are ordered with rotation varying fastest: index ->
        (rotation = index % 4, col = index // 4).
        """
        idx = int(index)
        rot = idx % 4
        col = idx // 4
        return (rot, col)

    def action_space(self):
        """Return a simple description of the action space.

        This is a discrete integer action space of size `4 * cols`. Each
        integer maps to a (rotation, column) pair via
        `action_index_to_pair(index)`.
        """
        return {'type': 'discrete', 'n': self.action_count(), 'mapping': 'rotation=index%4, col=index//4'}

    def get_state_space(self):
        """Return a description of the state/observation space.

        The environment state is returned by `get_state()` as `(board, piece)`:
          - `board` is a 2D list of shape `(rows, cols)` with 0/1 entries
          - `piece` is one of the available piece keys
        This method returns a concise specification for downstream use.
        """
        return {
            'board': {'shape': (self.rows, self.cols), 'dtype': 'int', 'values': [0, 1]},
            'piece': {'choices': list(self.available_pieces)}
        }

    def state_space(self):
        """Alias for `get_state_space()`."""
        return self.get_state_space()

    def step(self, rotation, col=None):
        """Apply action (rotation, col) for the environment's current piece.

        Returns a tuple `(state, reward, terminated, truncated, info)` where:
            - `state` is the new environment state `(board, piece)` after placement
            - `reward` is number of lines cleared (0 if no placement)
            - `terminated` is True when placement immediately collides (game over)
            - `truncated` is always False here
            - `info` contains `{'invalid_action': True}` when the action was invalid and had no effect
        """
        # support calling as step(action_index) or step(rotation, col)
        if col is None and isinstance(rotation, int):
            # single-index action
            action_index = int(rotation)
            rot_idx, col_idx = self.action_index_to_pair(action_index)
            rotation = rot_idx
            col = col_idx

        piece = self.current_piece
        # choose rotation modulo the piece's available rotations
        base = self.PIECES[piece]
        rotations = self._rotations(base)
        ri = int(rotation) % max(1, len(rotations))
        rot = rotations[ri]

        # check column validity for this rotation; if invalid, treat as no-op
        max_x = max(x for x, y in rot)
        width = max_x + 1
        if not (0 <= col <= self.cols - width):
            # invalid action: no effect, flagged in info
            return self.get_state(), 0, False, False, {'invalid_action': True}

        # if collides immediately at row 0 -> game over (valid action that ends episode)
        if self._would_collide(rot, 0, col):
            return self.get_state(), 0, True, False, {'invalid_action': False}

        # perform placement (same semantics as place_piece but without raising)
        row = 0
        while not self._would_collide(rot, row + 1, col):
            row += 1

        for x, y in rot:
            r = row + y
            c = col + x
            self.grid[r][c] = 1

        removed = self._clear_full_lines()
        self.last_cleared_lines = removed
        reward = getattr(self, 'last_cleared_lines', 0)
        # spawn next piece and return state including that next piece
        self.spawn_piece()
        return self.get_state(), reward, False, False, {'invalid_action': False}

    def pretty_print(self):
        for r in range(self.rows):
            print(''.join('#' if c else '.' for c in self.grid[r]))
