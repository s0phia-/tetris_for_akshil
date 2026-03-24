import unittest
import numpy as np
from tetris import TetrisEnv

class TestTetrisFeatures(unittest.TestCase):
    def setUp(self):
        # Using 4x4 for most tests as it's easy to manually verify
        self.env = TetrisEnv(rows=4, cols=4, piece_types=1) # Only 'I' piece

    def test_empty_board_features(self):
        """Empty board should have specific baseline transitions and 0 holes/wells."""
        empty_board = [[0]*4 for _ in range(4)]
        feats = self.env.add_node_features(empty_board)
        
        self.assertEqual(feats['row_transitions'], 8)
        self.assertEqual(feats['col_transitions'], 4)
        self.assertEqual(feats['holes'], 0)
        self.assertEqual(feats['cumulative_wells'], 0)

    def test_deep_well_logic(self):
        """A well of depth 3 should result in 1+2+3=6 cumulative depth."""
        # Col 1 is a well of depth 3
        board = [
            [1, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 1],
            [0, 0, 0, 0]
        ]
        feats = self.env.add_node_features(board)
        self.assertEqual(feats['cumulative_wells'], 6)

    def test_hole_detection(self):
        """Test hole count and hole depth (blocks above hole)."""
        # Two blocks in col 0, with a hole at the bottom
        board = [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        feats = self.env.add_node_features(board)
        self.assertEqual(feats['holes'], 2)
        # Hole at index (2,0) has 2 blocks above it. Hole at (3,0) has 2 blocks above it. Total 4.
        self.assertEqual(feats['hole_depth'], 4)
        self.assertEqual(feats['rows_with_holes'], 2)

    def test_landing_height(self):
        """Landing height should be the average height of placed blocks from the bottom."""
        board = [[0]*4 for _ in range(4)]
        # Piece occupies bottom two rows (indices 2 and 3)
        # Heights from bottom are 2 and 1. (2 + 1) / 2 = 1.5
        feats = self.env.add_node_features(board, placed_rows=[2, 3])
        self.assertEqual(feats['landing_height'], 1.5)

class TestTetrisMechanics(unittest.TestCase):
    def setUp(self):
        self.env = TetrisEnv(rows=4, cols=4, piece_types=['I'])

    def test_line_clear(self):
        """Simulate filling a row with a horizontal piece and ensure it clears completely."""
        # Start with an empty board
        board = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
        
        # Use Rotation 0 (Horizontal 'I') at Column 0
        # This piece is 4 cells wide, so it fills the entire 4x4 row.
        new_board, collided, lines, _ = self.env._simulate_place(board, 'I', 0, 0)
        
        self.assertFalse(collided, "Horizontal I-piece should fit on an empty 4x4 board.")
        self.assertEqual(lines, 1, "The row should have been cleared.")
        
        # Since the piece was only 1 row high, the board should be empty again
        self.assertEqual(sum(new_board[3]), 0, "Row 3 should be empty after the clear.")

    def test_collision(self):
        """Placing a piece where it cannot fit should trigger collision."""
        # Top row is already full
        board = [[1,1,1,1], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
        _, collided, _, _ = self.env._simulate_place(board, 'I', 0, 0)
        self.assertTrue(collided)

class TestSTGGeneration(unittest.TestCase):
    def test_stg_structure(self):
        """Generate a tiny STG and verify graph properties."""
        # Minimal setup: 3x3 board, 1 piece type
        env = TetrisEnv(rows=3, cols=3, piece_types=1)
        stg = env.generate_stg(directed=True)
        
        # 1. Check if terminal nodes exist
        terminals = [n for n, d in stg.nodes(data=True) if d.get('is_terminal') == 1]
        self.assertGreater(len(terminals), 0, "STG should have at least one terminal node")
        
        # 2. Check if features are attached to nodes
        first_node = list(stg.nodes(data=True))[0][1]
        self.assertIn('holes', first_node)
        self.assertIn('row_transitions', first_node)
        self.assertIn('piece', first_node)


class TestTetrisFeatures(unittest.TestCase):
    def setUp(self):
        # 4x4 is ideal for manual verification
        self.env = TetrisEnv(rows=4, cols=4)

    def test_baseline_empty_board(self):
        """Tests transitions on a totally empty board."""
        board = np.zeros((4, 4), dtype=int)
        feats = self.env.add_node_features(board)
        
        # Row: 1 [0 0 0 0] 1 -> 2 transitions per row * 4 rows = 8
        self.assertEqual(feats['row_transitions'], 8)
        # Col: 0 [0 0 0 0] 1 -> 1 transition per col * 4 cols = 4
        self.assertEqual(feats['col_transitions'], 4)
        # On an empty board with full boundaries, every cell is technically a well
        # Col 0: 1+2+3+4=10. Total for 4 columns = 40.
        self.assertEqual(feats['cumulative_wells'], 0)
        self.assertEqual(feats['holes'], 0)

    def test_hole_logic(self):
        """Tests holes and hole depth (blocks above holes)."""
        board = [
            [0, 1, 0, 0], # Row 0: Block at col 1
            [0, 0, 0, 0], # Row 1: Hole at col 1
            [0, 1, 0, 0], # Row 2: Block at col 1
            [0, 0, 0, 0]  # Row 3: Hole at col 1
        ]
        feats = self.env.add_node_features(board)
        
        # Column 1 has two holes (indices 1,1 and 3,1)
        self.assertEqual(feats['holes'], 2)
        # Hole at (1,1) has 1 block above it.
        # Hole at (3,1) has 2 blocks above it (Row 0 and Row 2).
        # Total depth: 1 + 2 = 3
        self.assertEqual(feats['hole_depth'], 3)
        self.assertEqual(feats['rows_with_holes'], 2)

    def test_transition_boundaries(self):
        """Tests that row boundaries are Full and col boundaries are Empty(top)/Full(bottom)."""
        # A completely full board
        board = np.ones((4, 4), dtype=int)
        feats = self.env.add_node_features(board)
        
        # Row: 1 [1 1 1 1] 1 -> 0 transitions
        self.assertEqual(feats['row_transitions'], 0)
        # Col: 0 [1 1 1 1] 1 -> Only 1 transition at the very top (0 to 1)
        # 1 trans * 4 columns = 4
        self.assertEqual(feats['col_transitions'], 4)

    def test_cumulative_wells(self):
        """Tests deep well calculation (1+2+3...)."""
        # Column 1 is a well of depth 3. Column 3 is a well of depth 4 (edge).
        board = [
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 0]
        ]
        feats = self.env.add_node_features(board)
        
        # Col 1: (1+2+3) = 6
        # Col 3: (1+2+3+4) = 10
        # Total = 16
        self.assertEqual(feats['cumulative_wells'], 16)

    def test_landing_height(self):
        """Tests (y1 + y2) / 2 height calculation."""
        # Assume a 4x4 board. Row 0 is height 4, Row 3 is height 1.
        # Placing a piece that occupies Row 1 and Row 2.
        # Heights: 3 and 2. Average: (3 + 2) / 2 = 2.5
        feats = self.env.add_node_features(np.zeros((4,4)), placed_rows=[1, 2])
        self.assertEqual(feats['landing_height'], 2.5)


if __name__ == '__main__':
    unittest.main()
