# Dependencies
import os
import yaml
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT_DIR)
sys.path.append('code')
from constants import RANDOM_SEED

# Config
logging.basicConfig(level=logging.INFO)

sns.set_style(style='whitegrid')
sns.set_context('notebook')

# Loading data
with open(os.path.join('code', 'rule_based', 'dialogue_act_rules.yml'), 'r') as file:
    rules = yaml.safe_load(file)

df_dialog_acts = pd.read_csv(os.path.join('data', 'dialog_acts.csv'))

# Pre-processing
df_dialog_acts.dropna(inplace=True)

## Normal
df_dialog_acts['utterance'] = df_dialog_acts['utterance'].apply(str.lower)
df_dialog_acts['act'] = df_dialog_acts['act'].apply(str.lower)

X_train, X_test, y_train, y_test = train_test_split(df_dialog_acts['utterance'], df_dialog_acts['act'],
                                                    test_size=0.15, 
                                                    random_state=RANDOM_SEED,
                                                    shuffle=True)

# Only 'train' on the training data by avoiding insights on test set. 
df_train = pd.DataFrame({
    'utterance': X_train,
    'act': y_train
})

df_dialog_acts_deduplicated = df_dialog_acts.drop_duplicates(subset='utterance')


# Data insights
logging.info(f"Classes: {df_dialog_acts['act'].unique()}")
logging.info(f"Value Counts: {df_dialog_acts['act'].value_counts()}")

def plot_act_distribution(df: pd.DataFrame, save_name: str, show_fig: bool, data_name: str):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='act', order = df['act'].value_counts().index)
    plt.xlabel('Dialogue Act')
    plt.ylabel('Turn Frequency')
    plt.title(f'Distribution of Dialogue Acts - {data_name} Data')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join('assets', save_name))
    if show_fig:
        plt.show()

plot_act_distribution(df_train, 'dialogue_act_distribution_train.png', show_fig=False, data_name='Full')
plot_act_distribution(df_dialog_acts, 'dialogue_act_distribution_full.png', show_fig=False, data_name='Full')
plot_act_distribution(df_dialog_acts_deduplicated, 'dialogue_act_distribution_deduplicated.png', show_fig=False, data_name='Deduplicated')
