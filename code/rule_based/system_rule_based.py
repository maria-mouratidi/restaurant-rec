import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT_DIR)
sys.path.append('code')
from constants import MAJORITY_CLASS, RANDOM_SEED, RULES
from model_evaluation import evaluate_model


def evaluate_rule(utterance_input: str, rule: dict) -> float:
    """Evaluates a single rule - utterance pair and returns the score.

    Args:
        utterance_input (str): utterance to be classified
        rule (dict): a certain rule with one or more of the following word matching lists: must_have, must_not_have, optional

    Returns:
        int: Corresponding score given the rule and utterance input.
    """    
    must_have = rule.get('must_have', [])
    must_not_have = rule.get('must_not_have', [])
    optional = rule.get('optional', [])
    
    score = 0.0

    if all(keyword in utterance_input for keyword in must_have) and all(keyword not in utterance_input for keyword in must_not_have):
        score += 1  

        for keyword in optional:
            if keyword in utterance_input:
                score += .5 

        return score
    return 0.0


def classify_rule_based(utterance_input: str) -> Tuple[str, Optional[str]]:
    """Classify a given utterance based on hard-coded rules and a scoring system.

    Args:
        utterance_input (str): utterance to be classified

    Returns:
        (str, str): Either the majority class if no rules matched or the 2 acts corresponding to the highest scored rules.
    """    
    act_scores = []
    for act, rule_list in RULES.items():
        for rule in rule_list:
            score = evaluate_rule(utterance_input, rule)
            if score > 0:
                act_scores.append((act, score))
    act_scores = sorted(act_scores, key=lambda x: x[1], reverse=True)

    if len(act_scores) < 1:
        return MAJORITY_CLASS, None
    elif len(act_scores) == 1:
        return act_scores[0][0], None
    else:
        return act_scores[0][0], act_scores[1][0]
    

def evaluate_rule_based(df_dialog_acts: pd.DataFrame, df_dialog_acts_deduplicated: pd.DataFrame):
    """Evaluation call for feed forward system, will print the metrics on the test set for both the full and
    deduplicated dataset"""
    for df_dialog_acts_version, df_version_name in [(df_dialog_acts, 'full'), (df_dialog_acts_deduplicated, 'deduplicated')]:
        _, X_test, _, y_test = train_test_split(df_dialog_acts_version['utterance'], df_dialog_acts_version['act'],
                                                            test_size=0.15,
                                                            random_state=RANDOM_SEED,
                                                            shuffle=True)

        # Inference
        rb_predictions = X_test.apply(classify_rule_based)

        # Evaluation
        print(f'Evaluation results for: {df_version_name} data ==============================\n')
        evaluate_model(y_true=y_test, y_pred=rb_predictions, system='rule_based', type=df_version_name, eval_top_2=True)
    