import pandas as pd
import itertools
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)

chars = 'AB'
length = 3
rows = []

for i in range(8):
    permutations = itertools.product(chars, repeat=length)

    for p in permutations:
        if len(set(p)) == 1:
            continue
        row = list(p)
        rows.append(row)

df = pd.DataFrame(rows, columns=['condition_task_1', 'condition_task_2', 'condition_task_3'])
df['participant_number'] = range(1, len(df) + 1)

cols = ['participant_number'] + [col for col in df if col != 'participant_number']
df = df[cols]

print(df)
df.to_csv(os.path.join('data', 'participant_conditions.csv'), index=False)
