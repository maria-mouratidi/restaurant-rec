# Dependencies
import os
import logging
import pickle
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from keras_tuner import RandomSearch

import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT_DIR)
sys.path.append('code')
from constants import RANDOM_SEED
from model_evaluation import prepare_data

# Config
logging.basicConfig(level=logging.INFO)
RERUN_HP_TUNING = False

df_dialog_acts, df_dialog_acts_deduplicated = prepare_data()

# Embedding model
embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# Training and saving for both versions of data
for df_dialog_acts_version, df_version_name in [(df_dialog_acts, 'full'), (df_dialog_acts_deduplicated, 'deduplicated')]:
    ## Normal
    X_train_val, X_test, y_train_val, y_test = train_test_split(df_dialog_acts_version['utterance'], df_dialog_acts_version['act'],
                                                        test_size=0.15,
                                                        random_state=RANDOM_SEED,
                                                        shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                    test_size=0.15,
                                                    random_state=RANDOM_SEED,
                                                    shuffle=True)

    logging.info(f'X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}')
    logging.info(f'y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}')

    # Create embeddings
    X_train_embed = embedding_model.encode(X_train.tolist(), normalize_embeddings=True)
    X_val_embed = embedding_model.encode(X_val.tolist(), normalize_embeddings=True)
    X_test_embed = embedding_model.encode(X_test.tolist(), normalize_embeddings=True)

    # Encoding Y labels
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    y_val_encoded = encoder.transform(y_val)

    def build_model(hp):
        input_shape = X_train_embed.shape[1:]  # shape of a single embedding
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(units=hp.Int('units1', min_value=16, max_value=512, step=32),
                                  activation=hp.Choice('activation1', ['relu', 'tanh', 'sigmoid'])),
            Dropout(hp.Float('dropout1', min_value=0.0, max_value=0.5, step=0.1)),
            tf.keras.layers.Dense(units=hp.Int('units2', min_value=64, max_value=512, step=32),
                                  activation=hp.Choice('activation2', ['relu', 'tanh', 'sigmoid'])),
            Dropout(hp.Float('dropout2', min_value=0.0, max_value=0.5, step=0.1)),
            tf.keras.layers.Dense(units=hp.Int('units3', min_value=32, max_value=512, step=32),
                                  activation=hp.Choice('activation3', ['relu', 'tanh', 'sigmoid'])),
            tf.keras.layers.Dense(len(y_train.unique()), activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=100,
        executions_per_trial=3,
        directory=os.path.join('code', 'feed_forward', f'{df_version_name}_model'),
        project_name='random_search_tuning')

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    if RERUN_HP_TUNING:
        tuner.search(X_train_embed, y_train_encoded,
                     epochs=10,
                     batch_size=32,
                     validation_data=(X_val_embed, y_val_encoded),
                     callbacks=[early_stopping])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        with open(os.path.join('code', 'feed_forward', f'{df_version_name}_model', 'best_ff_parameters.pkl'), "wb") as f:
            pickle.dump(best_hps, f)

    with open(os.path.join('code', 'feed_forward', f'{df_version_name}_model', 'best_ff_parameters.pkl'), "rb") as f:
        best_hps = pickle.load(f)

    if RERUN_HP_TUNING:
        with open(os.path.join('code', 'feed_forward', f'{df_version_name}_model', 'best_ff_parameters.txt'), mode='w') as f:
            for hp, value in best_hps.values.items():
                f.write(f"{hp}: {value}\n")

    model = tuner.hypermodel.build(best_hps)

    early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    history = model.fit(X_train_embed, y_train_encoded,
                        epochs=30,
                        batch_size=32,
                        verbose=2,
                        validation_data=(X_val_embed, y_val_encoded),
                        callbacks=[early_stopping])

    # Plot of training vs Validation loss
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(len(history.history['loss'])), y=history.history['loss'], label='Training')
    sns.lineplot(x=range(len(history.history['val_loss'])), y=history.history['val_loss'], label='Validation', linestyle='--')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(title='Loss Types')
    plt.savefig(os.path.join('assets', f'training_validation_loss_{df_version_name}.png'))
    plt.show()

    # Save model, embedding model and encoder
    model.save(os.path.join('code', 'feed_forward', f'{df_version_name}_model'))

    with open(os.path.join('code', 'feed_forward', f'{df_version_name}_model', f'{df_version_name}_encoder.pkl'), "wb") as f:
        pickle.dump(encoder, f)
