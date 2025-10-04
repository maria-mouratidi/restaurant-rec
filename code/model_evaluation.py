import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
import numpy as np
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)

# Info:
# https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd

def prepare_data():
    df_dialog_acts = pd.read_csv(os.path.join('data', 'dialog_acts.csv'))

    # Pre-processing
    df_dialog_acts.dropna(inplace=True)
    df_dialog_acts['utterance'] = df_dialog_acts['utterance'].apply(str.lower)
    df_dialog_acts['act'] = df_dialog_acts['act'].apply(str.lower)

    df_dialog_acts_deduplicated = df_dialog_acts.drop_duplicates(subset='utterance')

    return df_dialog_acts, df_dialog_acts_deduplicated


def evaluate_model(y_true: pd.Series, y_pred: pd.Series, system: str, type: str, eval_top_2: bool) -> None:
    """
    Evaluates the model using several metrics and plots the confusion matrix.

    Args:
        y_true: A series of tuples with item1 and item2 being the best and second best guess respectively. 
        y_pred: The predicted label series.
        system: Label of current classification system
        type: on what type of data the metrics are to be disclosed
    """
    metrics_file_path = os.path.join('evaluation', system, f'metrics_{system}_{type}.txt')
    with open(metrics_file_path, 'w') as f:
        if eval_top_2:
            # Accuracy accounting for top 2 predicted classes
            top_2_count = 0
            for i, ground_truth in enumerate(y_true):
                if ground_truth in y_pred.iloc[i]:
                    top_2_count += 1
            top_2_accuracy = top_2_count / len(y_true)
            f.write(f'Top-2 Accuracy: {top_2_accuracy}\n\n')

            # Single prediction metrics
            y_pred = y_pred.apply(lambda t: t[0])

        # Single prediction metrics
        class_labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))

        accuracy = accuracy_score(y_true, y_pred)
        f.write(f'Accuracy: {accuracy}\n')
        
        kappa = cohen_kappa_score(y_true, y_pred)
        f.write(f'Cohen\'s Kappa: {kappa}\n')

        cr = classification_report(y_true, y_pred, target_names=class_labels, digits=3, zero_division=1)
        f.write(f'Classification Report:\n{cr}\n')

        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(15, 15))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis',
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix: {system}_{type}')
        plt.savefig(os.path.join('evaluation', system, f'cm_{system}_{type}.jpg'))

    with open(metrics_file_path, 'r') as f:
        print(f.read())
