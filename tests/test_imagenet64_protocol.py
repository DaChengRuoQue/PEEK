from __future__ import annotations

import dataclasses
import unittest

from peek.imagenet64.protocol import EvaluationProtocol, model_call_count, paper_protocol


class Imagenet64ProtocolTests(unittest.TestCase):
    def test_paper_protocol_has_one_fixed_compute_matched_pair(self) -> None:
        protocol = paper_protocol()

        self.assertEqual(protocol.num_images, 50_000)
        self.assertEqual(protocol.baseline_nfe, 9)
        self.assertEqual(protocol.peek_base_nfe, 6)
        self.assertEqual(protocol.peek_gates, (1, 1, 1, 0, 0))
        self.assertEqual(protocol.peek_total_nfe, 9)

    def test_model_call_count_counts_only_active_lookaheads(self) -> None:
        self.assertEqual(model_call_count(6, (1, 1, 1, 0, 0)), 9)
        self.assertEqual(model_call_count(9, ()), 9)

    def test_protocol_rejects_evaluation_size_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "50,000"):
            EvaluationProtocol(num_images=5_000)

    def test_protocol_rejects_gate_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed PEEK gate"):
            EvaluationProtocol(peek_gates=(1, 1, 0, 0, 0))

    def test_protocol_is_immutable(self) -> None:
        protocol = paper_protocol()

        with self.assertRaises(dataclasses.FrozenInstanceError):
            protocol.num_images = 1  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
