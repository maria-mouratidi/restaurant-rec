import os
import pickle
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT_DIR)
sys.path.append('code')
from constants import RANDOM_SEED
from model_evaluation import evaluate_model


def dt_load_mod_vec_enc(model_version: str) -> Tuple[object, object, object]:
    """Load the model, vectorizer and encoder for inference with the trained decision tree model of supplied type."""
    # Load the decision tree model, vectorizer, and encoder
    with open(os.path.join('code', 'decision_tree', f'{model_version}_model', f'{model_version}_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join('code', 'decision_tree', f'{model_version}_model', f'{model_version}_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join('code', 'decision_tree', f'{model_version}_model', f'{model_version}_encoder.pkl'), 'rb') as f:
        encoder = pickle.load(f)

    return model, vectorizer, encoder


def classify_decision_tree(utterance_input: str, model, vectorizer, encoder) -> str:
    """Takes the input utterance and classifies it using the supplied model dependencies."""
    user_input_vec = vectorizer.transform([utterance_input])
    prediction = model.predict(user_input_vec)
    label = encoder.inverse_transform(prediction)

    return label[0]


def evaluate_decision_tree(df_dialog_acts: pd.DataFrame, df_dialog_acts_deduplicated: pd.DataFrame):
    """Evaluation call for decision tree system, will print the metrics on the test set for both the full and
    deduplicated dataset"""
    # Evaluation for both versions of data
    for df_dialog_acts_version, df_version_name in [(df_dialog_acts, 'full'), (df_dialog_acts_deduplicated, 'deduplicated')]:
        _, X_test, _, y_test = train_test_split(df_dialog_acts_version['utterance'],
                                                            df_dialog_acts_version['act'],
                                                            test_size=0.15,
                                                            random_state=RANDOM_SEED,
                                                            shuffle=True)

        print(f'Evaluation results for: {df_version_name} data ==============================\n')
        model_deps = dt_load_mod_vec_enc(df_version_name)
        dt_predictions = X_test.apply(lambda x: classify_decision_tree(x, *model_deps))
        evaluate_model(y_true=y_test, y_pred=dt_predictions, system='decision_tree', type=df_version_name, eval_top_2=False)
