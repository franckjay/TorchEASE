import unittest
import pandas as pd
import torch
from main.EASE import TorchEASE


class TestBasicFunction(unittest.TestCase):
    def setUp(self):

        self.df = pd.DataFrame(
            {
                "user": [
                    "jay",
                    "jay",
                    "jay",
                    "esther",
                    "esther",
                    "harald",
                    "harald",
                    "harald",
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
                "score": [5, 4.3, 2, 1, 5, 3.5, 3.5, 2.0],
            }
        )

    def test_torch_implicit(self):
        self.tei = TorchEASE(self.df, user_col="user", item_col="item", reg=0.05)
        self.tei.fit()
        self.assertEqual(
            round(self.tei.B[0].sum().item(), 3),
            round(
                torch.FloatTensor([0.00, 0.1296, 0.1296, -0.4540, 0.6806, 0.4878])
                .sum()
                .item(),
                3,
            ),
        )
        print (self.tei.sparse.to_dense()[0] @ self.tei.B)
        self.assertTrue(
            torch.argmax(self.tei.sparse.to_dense()[0] @ self.tei.B) == torch.tensor(3)
        )

    def test_torch_explicit(self):
        self.tee = TorchEASE(
            self.df, user_col="user", item_col="item", score_col="score", reg=0.05
        )
        self.tee.fit()
        self.assertTrue(
            torch.argmax(self.tee.sparse.to_dense()[0] @ self.tee.B), torch.tensor(5)
        )


if __name__ == "__main__":
    unittest.main()
