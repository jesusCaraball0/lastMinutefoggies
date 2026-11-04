"""King-safety regressions for observation-only FOW move selection."""
import unittest

import chess

from referee import FowBoard
from starters.shared.moves import candidates, ObservationTracker


def view_of(board):
    return ''.join(board.piece_at(square).symbol() if board.piece_at(square) else '-'
                   for square in chess.SQUARES)


def choices(board, view=None, previous=None):
    visible, moves = candidates(view or view_of(board), board.turn,
                                board.castling_rights, previous)
    # Simulating candidates must leave the returned board untouched.
    assert visible.turn == board.turn and not visible.move_stack
    return {move.uci() for move in moves}


class KingSafetyTests(unittest.TestCase):
    def test_escape_block_and_capture_for_both_colors(self):
        board = FowBoard('4r2k/8/8/8/8/8/R7/4K3 w - - 0 1')
        for position in (board, board.mirror()):
            moves = choices(position)
            transform = (lambda square: square) if position.turn else chess.square_mirror
            def uci(text):
                move = chess.Move.from_uci(text)
                return chess.Move(transform(move.from_square), transform(move.to_square)).uci()
            self.assertIn(uci('a2e2'), moves)  # Block the rook attack.
            self.assertIn(uci('e1d1'), moves)  # Escape with the king.
            self.assertNotIn(uci('a2a3'), moves)
        board = FowBoard('7k/8/8/8/8/8/4r3/4K3 w - - 0 1')
        self.assertIn('e1e2', choices(board))  # Capture an undefended attacker.

    def test_pinned_blocker_cannot_expose_king(self):
        board = FowBoard('4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1')
        moves = choices(board)
        self.assertNotIn('e2d2', moves)
        self.assertIn('e2e8', moves)

    def test_castling_checks_final_square_only(self):
        # FOW permits castling across an attacked square.
        board = FowBoard('k4r2/8/8/8/8/8/8/4K2R w K - 0 1')
        self.assertIn('e1g1', choices(board))
        board = FowBoard('k5r1/8/8/8/8/8/8/4K2R w K - 0 1')
        self.assertNotIn('e1g1', choices(board))

    def test_en_passant_cannot_expose_rook_attack(self):
        board = FowBoard('7k/8/8/r4pPK/8/8/8/8 w - f6 0 2')
        before = board.copy()
        before.remove_piece_at(chess.F5)
        before.set_piece_at(chess.F7, chess.Piece(chess.PAWN, chess.BLACK))
        self.assertNotIn('g5f6', choices(board, previous=view_of(before)))
        # Confirm EP was inferred: the visible candidate board retains f6.
        visible, _ = candidates(view_of(board), chess.WHITE, 0, view_of(before))
        self.assertEqual(visible.ep_square, chess.F6)

    def test_unknown_blocker_does_not_prove_slider_attack(self):
        board = FowBoard('4r2k/8/8/8/8/8/R7/4K3 w - - 0 1')
        view = list(view_of(board))
        view[chess.E4] = '?'
        self.assertIn('a2a3', choices(board, ''.join(view)))

    def test_unknown_piece_is_not_invented(self):
        board = FowBoard('7k/8/8/8/8/8/R7/4K3 w - - 0 1')
        view = list(view_of(board))
        view[chess.E8] = '?'
        self.assertIn('a2a3', choices(board, ''.join(view)))

    def test_no_safe_move_retains_legal_fallback(self):
        board = FowBoard('7R/8/8/8/8/2k5/1q6/K7 w - - 0 1')
        # The queen attacks a1, a2, b1; the black king defends b2.
        moves = choices(board)
        self.assertIn('h8h7', moves)
        self.assertTrue(moves)
        for uci in moves:
            board.push_uci(uci)
            self.assertTrue(board.is_attacked_by(chess.BLACK, board.king(chess.WHITE)))
            board.pop()

    def test_tracker_uses_safety_filter(self):
        board = FowBoard('4r2k/8/8/8/8/8/R7/4K3 w - - 0 1')
        tracker = ObservationTracker(chess.WHITE, rights=0)
        tracker.observe(view_of(board))
        self.assertNotIn(chess.Move.from_uci('a2a3'), tracker.legal_moves())


if __name__ == '__main__':
    unittest.main()
