import unittest
import pandas as pd
from main.metrics import hit_rate_k


class TestBasicMetrics(unittest.TestCase):
    def setUp(self):
        self.all_misses = pd.DataFrame(
            {
                "predicted": [
                    ["jay"],
                    ["jay"],
                    ["jay"],
                    ["esther"],
                    ["esther"],
                    ["harald"],
                    ["harald"],
                    ["harald"],
                ],
                "item": [
                    "The Matrix",
                    "Arrival",
                    "Wall-E",
                    "Mulan",
                    "Wall-E",
                    "Mulan",
                    "Black Beauty",
                    "Incredibles",
                ],
            }
        )

        self.all_hits = pd.DataFrame(
            {
                "predicted": [
                    ["The Matrix"],
                    ["Arrival"],
                    ["Wall-E"],
                    ["Mulan"],
                    ["Wall-E"],
                    ["Mulan"],
                    ["Black Beauty"],
                    ["Incredibles"],
                ],
                "item": [
                    "The Matrix",
                    "Arrival",
                    "Wall-E",
                    "Mulan",
                    "Wall-E",
                    "Mulan",
                    "Black Beauty",
                    "Incredibles",
                ],
            }
        )

        self.partial_hits = pd.DataFrame(
            {
                "predicted": [
                    ["The Shining", "Blade Runner"],
                    ["Oh Brother, Where art Thou?", "Arrival"],
                    ["The Lion King", "Harry Potter"],
                    ["Mulan", "Cats"],
                    ["Wall-E", "Wall-E"],
                    ["Mulan", "Wall-E"],
                    ["Tron", "Tron Legacy"],
                    ["Incredibles", "Incredibles 2"],
                ],
                "item": [
                    "The Matrix",
                    "Arrival",
                    "Wall-E",
                    "Mulan",
                    "Wall-E",
                    "Mulan",
                    "Black Beauty",
                    "Incredibles",
                ],
            }
        )

    def test_all_hits(self):
        self.assertTrue(
            hit_rate_k(self.all_hits, actual_col="item", pred_col="predicted") == 1.0
        )

    def test_all_misses(self):
        self.assertTrue(
            hit_rate_k(self.all_misses, actual_col="item", pred_col="predicted") == 0.0
        )

    def test_partial_misses(self):
        self.assertTrue(
            hit_rate_k(self.partial_hits, actual_col="item", pred_col="predicted")
            == 5./8.
        )


if __name__ == "__main__":
    unittest.main()
