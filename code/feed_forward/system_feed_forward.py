import os
import pickle
import numpy as np
import pandas as pd
import sys
from typing import Tuple
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT_DIR)
sys.path.append('code')
from constants import RANDOM_SEED
from model_evaluation import evaluate_model


tqdm.pandas()

def ff_load_mod_vec_enc(df_version_name: str):
    """Load the model, embedding model and encoder for inference with the feed forward model of supplied type."""
    model = load_model(os.path.join('code', f'feed_forward', f'{df_version_name}_model'))

    embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')

    with open(os.path.join('code', 'feed_forward', f'{df_version_name}_model', f'{df_version_name}_encoder.pkl'),
              "rb") as f:
        encoder = pickle.load(f)

    return model, embedding_model, encoder


def classify_feed_forward(utterance_input: str, model, embedding_model, encoder) -> Tuple[str, str]:
    """Takes the input utterance and classifies it using the supplied model dependencies."""
    user_input_embed = embedding_model.encode([utterance_input], normalize_embeddings=True, show_progress_bar=False)

    prediction = model.predict(user_input_embed, verbose=0)
    top2_indices = np.argsort(prediction[0])[-2:][::-1]
    top2_labels = encoder.inverse_transform(top2_indices)

    return top2_labels[0], top2_labels[1]


def evaluate_feed_forward(df_dialog_acts: pd.DataFrame, df_dialog_acts_deduplicated: pd.DataFrame):
    """Evaluation call for feed forward system, will print the metrics on the test set for both the full and
    deduplicated dataset"""
    print(f"{os.cpu_count()} cores being used for inference.")

    # Evaluation for both versions of data
    for df_dialog_acts_version, df_version_name in [(df_dialog_acts, 'full'), (df_dialog_acts_deduplicated, 'deduplicated')]:
        _, X_test, _, y_test = train_test_split(df_dialog_acts_version['utterance'], df_dialog_acts_version['act'],
                                                        test_size=0.15,
                                                        random_state=RANDOM_SEED,
                                                        shuffle=True)


        print(f'Evaluation results for: {df_version_name} data ==============================\n')
        model_deps = ff_load_mod_vec_enc(df_version_name)

        progress_bar = tqdm(total=len(X_test), desc="Inference", dynamic_ncols=True)

        def tqdm_classify_feed_forward(x):
            result = classify_feed_forward(x, *model_deps)
            progress_bar.update(1)
            return result

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(tqdm_classify_feed_forward, X_test))
            ff_predictions = pd.Series(results)

        progress_bar.close()
        evaluate_model(y_true=y_test, y_pred=ff_predictions, system='feed_forward', type=df_version_name, eval_top_2=True)
