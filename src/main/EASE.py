import sys
import logging

import torch
import pandas as pd


class TorchEASE:
    def __init__(
        self, train, user_col="user_id", item_col="item_id", score_col=None, reg=250.0
    ):
        """

        :param train: Training DataFrame of user, item, score(optional) values
        :param user_col: Column name for users
        :param item_col: Column name for items
        :param score_col: Column name for scores. Implicit feedback otherwise
        :param reg: Regularization parameter.
                    Change by orders of magnitude to tune (2e1, 2e2, ...,2e4)
        """
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
        )

        self.logger = logging.getLogger("notebook")
        self.logger.info("Building user + item lookup")
        # How much regularization do you need?
        self.reg = reg

        self.user_col = user_col
        self.item_col = item_col

        self.user_id_col = user_col + "_id"
        self.item_id_col = item_col + "_id"

        self.user_lookup = self.generate_labels(train, self.user_col)
        self.item_lookup = self.generate_labels(train, self.item_col)

        self.item_map = {}
        self.logger.info("Building item hashmap")
        for _item, _item_id in self.item_lookup.values:
            self.item_map[_item_id] = _item

        train = pd.merge(train, self.user_lookup, on=[self.user_col])
        train = pd.merge(train, self.item_lookup, on=[self.item_col])
        self.logger.info("User + item lookup complete")
        self.indices = torch.LongTensor(
            train[[self.user_id_col, self.item_id_col]].values
        )

        if not score_col:
            # Implicit values only
            self.values = torch.ones(self.indices.shape[0])
        else:
            # TODO: Test if score_col works correctly
            self.values = torch.FloatTensor(train[score_col])

        # TODO: Is Sparse the best implementation?
        self.sparse = torch.sparse.FloatTensor(self.indices.t(), self.values)

        self.logger.info("Sparse data built")

    def generate_labels(self, df, col):
        dist_labels = df[[col]].drop_duplicates()
        dist_labels[col + "_id"] = dist_labels[col].astype("category").cat.codes

        return dist_labels

    def fit(self):
        self.logger.info("Building G Matrix")
        G = self.sparse.to_dense().t() @ self.sparse.to_dense()
        G += torch.eye(G.shape[0]) * self.reg

        P = G.inverse()

        self.logger.info("Building B matrix")
        B = P / (-1 * P.diag())
        # Set diagonals to 0. TODO: Use .fill_diag_
        B = B + torch.eye(B.shape[0])

        # Predictions for user `_u` will be self.sparse.to_dense()[_u]@self.B
        self.B = B

        return

    def predict_all(self, pred_df, k=5, remove_owned=True):
        """
        :param pred_df: DataFrame of users that need predictions
        :param k: Number of items to recommend to each user
        :param remove_owned: Do you want previously interacted items included?
        :return: DataFrame of users + their predictions in sorted order
        """
        pred_df = pred_df[[self.user_col]].drop_duplicates()
        n_orig = pred_df.shape[0]

        # Alert to number of dropped users in prediction set
        pred_df = pd.merge(pred_df, self.user_lookup, on=[self.user_col])
        n_curr = pred_df.shape[0]
        if n_orig - n_curr:
            self.logger.info(
                "Number of unknown users from prediction data = %i" % (n_orig - n_curr)
            )

        _output_preds = []
        # Select only user_ids in our user data
        _user_tensor = self.sparse.to_dense().index_select(
            dim=0, index=torch.LongTensor(pred_df[self.user_id_col])
        )

        # Make our (raw) predictions
        _preds_tensor = _user_tensor @ self.B
        self.logger.info("Predictions are made")
        if remove_owned:
            # Discount these items by a large factor (much faster than list comp.)
            self.logger.info("Removing owned items")
            _preds_tensor += -1.0 * _user_tensor

        self.logger.info("TopK selected per user")
        for _preds in _preds_tensor:
            # Very quick to use .topk() vs. argmax()
            _output_preds.append(
                [self.item_map[_id] for _id in _preds.topk(k).indices.tolist()]
            )

        pred_df["predicted_items"] = _output_preds
        self.logger.info("Predictions are returned to user")
        return pred_df

    def score_predictions(self):
        # TODO: Implement this with some common metrics
        return None
