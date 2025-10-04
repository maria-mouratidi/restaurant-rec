# Dependencies
import logging
import os

import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from collections import Counter
from tqdm import tqdm
from itertools import combinations
from sklearn.model_selection import train_test_split

import sys
sys.path.append('code')
from constants import RANDOM_SEED

# Config
logging.basicConfig(level=logging.INFO)


# Loading data
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

# Insights for manual evaluation
logging.info(f'Generating co-occurrences and wordclouds for all acts...')
co_occurrence_counters = {}

for act, df_group in tqdm(df_train.groupby('act')):
    concatenated_utterances = df_group['utterance'].str.cat(sep=' ')

    # Wordclouds
    act_wc = WordCloud(max_font_size=50, max_words=1000, background_color="white", stopwords=set()).generate(concatenated_utterances)
    plt.figure()
    plt.imshow(act_wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f'Act: {act}')
    plt.savefig(os.path.join('assets', 'wordclouds', f'wc_{act}.png'))

    # Co-occurrences
    co_occurrence_counters[act] = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter(), 5: Counter()}

    # Loop through the utterances in this group
    for _, row in df_group.iterrows():
        utterance = row['utterance']
        words = utterance.split()

        for r in [5, 4, 3, 2, 1]:
            word_combinations = combinations(words, r)
            co_occurrence_counters[act][r].update(word_combinations)

    with open(os.path.join('data', 'co_occurrences', f"co_occurrences_{act}.txt"), "w") as f:
        f.write(f"{act} {'='*50}\n")

        for r in [5, 4, 3, 2, 1]:
            f.write(f"\nCombinations of {r} Words {'-'*30}\n")

            for combo, count in co_occurrence_counters[act][r].most_common():
                if count > 1: 
                    f.write(f"{', '.join(combo)}: {count}\n")
