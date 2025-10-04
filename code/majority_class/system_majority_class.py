# Dependencies
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT_DIR)
sys.path.append('code')
from model_evaluation import evaluate_model
from constants import RANDOM_SEED, MAJORITY_CLASS


# Majority class sytem
def classify_majority_class(utterance_input: str) -> str:
    """Will 'classify' the utterance input with the constant Majority Class."""
    return MAJORITY_CLASS


def evaluate_majority_class(df_dialog_acts: pd.DataFrame, df_dialog_acts_deduplicated: pd.DataFrame):
    """Evaluation call for majority class system, will print the metrics on the test set for both the full and
    deduplicated dataset"""
    for df_dialog_acts_version, df_version_name in [(df_dialog_acts, 'full'), (df_dialog_acts_deduplicated, 'deduplicated')]:
        _, X_test, _, y_test = train_test_split(df_dialog_acts_version['utterance'], df_dialog_acts_version['act'],
                                                            test_size=0.15,
                                                            random_state=RANDOM_SEED,
                                                            shuffle=True)

        # Inference
        mc_predictions = X_test.apply(classify_majority_class)

        # Evaluation
        print(f'Evaluation results for: {df_version_name} data ==============================\n')
        evaluate_model(y_true=y_test, y_pred=mc_predictions, system='majority_class', type=df_version_name, eval_top_2=False)
