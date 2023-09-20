# TorchEASE
Implementation of Embarrassingly Shallow Autoencoders (Harald Steck) in PyTorch
```
* Adapted from: https://github.com/Darel13712/ease_rec 
* Paper: https://arxiv.org/abs/1905.03375
* Papers with Code: https://paperswithcode.com/sota/collaborative-filtering-on-million-song
```

This code is currently SOTA on the Million Song challenge on Papers With Code.
It utilizes a simple closed-form solution for a recommendation problem. 
This PyTorch implementation is much faster than the original Numpy version.


## To use:
### Implicit interactions: 
1. `te_implicit = TorchEASE(df, user_col="user", item_col="item")`
2. `te_implicit.fit()`
3. `predictions = te_implicity.predict_all(predict_df)`
### Explicit interactions:
1. `te_explicit = TorchEASE(df, user_col="user", item_col="item", score_col="rating")`
2. `te_explicit.fit()`
3. `predictions = te_explicit.predict_all(predict_df)`


## To use via CLI:
1. Put your data files into `data/`
2. From a terminal window, call `python train.py {TRAIN.csv} {TO_PRED.csv} {USER_COLUMN_NAME} {ITEM_COLUMN_NAME} {[optional]SCORE_COLUMN_NAME}`
3. Predictions are pushed to `predictions/output.csv`

Example usage:
```
python train.py training.csv users.csv username book_name book_score
```

Tuning the `regularization` parameter on scales from `1E2` - `1E3` seems to be effective 
