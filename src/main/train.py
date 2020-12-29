import sys
import logging
import pandas as pd


from main.EASE import TorchEASE

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger("notebook")

if __name__ == "__main__":
    try:
        train_csv_file = sys.argv[1]
        pred_csv_file = sys.argv[2]
        user_col = sys.argv[3]
        item_col = sys.argv[4]
        score_col = sys.argv[5]
        logger.info(
            "Files to run: "
            + train_csv_file + " "+ pred_csv_file
            + " with user_col: "
            + user_col
            + " item_col "
            + item_col
            + " and score column "
            + score_col
        )

    except Exception as e:
        logger.error("Not enough information")
        raise e

    input_dir = "../data/"
    output_dir = "../predictions/"

    train_df = pd.read_csv(input_dir+train_csv_file)
    pred_df = pd.read_csv(input_dir+pred_csv_file)
    output_file = "output.csv"

    if not score_col or score_col == "None":
        score_col = None
    logger.info("Training model")
    te = TorchEASE(train_df, user_col=user_col, item_col=item_col, score_col=score_col)
    te.fit()

    logger.info("Making predictions")
    output = te.predict_all(pred_df, k=5)
    output.to_csv(output_dir+output_file)

    logger.info("CSV saved.")