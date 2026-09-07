from __future__ import annotations

import unittest

from ov_wiki_baseline_benchmark.datasets.locomo import render_session, session_numbers


class LoCoMoTests(unittest.TestCase):
    def test_session_numbers_ignore_timestamps(self) -> None:
        conversation = {
            "session_2": [],
            "session_2_date_time": "later",
            "session_1": [],
            "session_1_date_time": "earlier",
            "speaker_a": "A",
            "speaker_b": "B",
        }
        self.assertEqual(session_numbers(conversation), [1, 2])

    def test_render_session_preserves_temporal_and_image_text(self) -> None:
        sample = {
            "sample_id": "conv-test",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1_date_time": "1 pm on 1 January, 2024",
                "session_1": [
                    {
                        "dia_id": "D1:1",
                        "speaker": "A",
                        "text": "Hello",
                        "blip_caption": "a red bicycle",
                        "query": "red bicycle",
                    }
                ],
            },
        }
        text = render_session(sample, 1)
        self.assertIn("Session date and time: 1 pm on 1 January, 2024", text)
        self.assertIn("[D1:1] A: Hello", text)
        self.assertIn("[Image caption] a red bicycle", text)
        self.assertIn("[Image search query] red bicycle", text)


if __name__ == "__main__":
    unittest.main()
