# Dependencies
import os
import logging
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT_DIR)
sys.path.append('code')
from constants import RANDOM_SEED
from model_evaluation import prepare_data


# Config
logging.basicConfig(level=logging.INFO)

df_dialog_acts, df_dialog_acts_deduplicated = prepare_data()


for df_dialog_acts_version, df_version_name in [(df_dialog_acts, 'full'), (df_dialog_acts_deduplicated, 'deduplicated')]:
    # Normal
    X_train_val, X_test, y_train_val, y_test = train_test_split(df_dialog_acts_version['utterance'],
                                                                df_dialog_acts_version['act'],
                                                                test_size=0.15,
                                                                random_state=RANDOM_SEED,
                                                                shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=0.15,
                                                      random_state=RANDOM_SEED,
                                                      shuffle=True)

    logging.info(f'X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}')
    logging.info(f'y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}')

    vectorizer = CountVectorizer()

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    hp_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_val_encoded = encoder.transform(y_val)
    y_test_encoded = encoder.transform(y_test)

    model = DecisionTreeClassifier(random_state=RANDOM_SEED)

    # -1 is all cores
    grid_search = GridSearchCV(estimator=model, param_grid=hp_grid,
                               cv=10, n_jobs=-1, verbose=2, scoring='accuracy')

    grid_search.fit(X_train_vec, y_train_encoded)

    best_hps = grid_search.best_params_
    with open(os.path.join('code', 'decision_tree', f'{df_version_name}_model', 'best_dt_parameters.txt'), mode='w') as f:
        f.write(str(best_hps))

    model = DecisionTreeClassifier(**best_hps, random_state=RANDOM_SEED)
    model.fit(X_train_vec, y_train_encoded)

    y_val_pred = model.predict(X_val_vec)
    y_test_pred = model.predict(X_test_vec)

    val_accuracy = accuracy_score(y_val_encoded, y_val_pred)
    test_accuracy = accuracy_score(y_test_encoded, y_test_pred)

    print(f"Validation accuracy: {val_accuracy}")
    print(f"Test accuracy: {test_accuracy}")

    with open(os.path.join('code', 'decision_tree', f'{df_version_name}_model', f'{df_version_name}_model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    with open(os.path.join('code', 'decision_tree', f'{df_version_name}_model', f'{df_version_name}_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    with open(os.path.join('code', 'decision_tree', f'{df_version_name}_model', f'{df_version_name}_encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)
