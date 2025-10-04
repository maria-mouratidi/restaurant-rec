import os
import random
import pandas as pd
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)


# Writing .dat to .csv and changing null for no_act ====================================================================
acts = []
utterances = []

with open(os.path.join('data', 'dialog_acts.dat'), 'r') as file:
    for line in file:
        line_split = line.split()
        acts.append(line_split[0])
        utterances.append(' '.join(line_split[1:]))

# Loading as Dataframe
df_dialog_acts = pd.DataFrame({
    'utterance': utterances,
    'act': acts,
})

# null -> no_act to avoid issues
df_dialog_acts['act'].replace('null', 'no_act', inplace=True)

df_dialog_acts.to_csv(os.path.join('data', 'dialog_acts.csv'), index=False)



# Adding reasoning to restaurantname ===================================================================================
restaurant_catalogue = pd.read_csv(os.path.join('data', 'restaurant_info.csv'))

quality_options = ['terrible', 'bad', 'okay', 'good', 'excellent']
crowdedness_options = ['busy', 'airy', 'quiet']
lengthofstay_options = ['short', 'average', 'long']

n_rests = restaurant_catalogue.shape[0]

qualities = random.choices(quality_options, k=n_rests)
crowdednesses = random.choices(crowdedness_options, k=n_rests)
lengthsofstay = random.choices(lengthofstay_options, k=n_rests)

restaurant_catalogue = restaurant_catalogue.assign(
    quality=qualities,
    crowdedness=crowdednesses,
    lengthofstay=lengthsofstay
)

# Save to file
restaurant_catalogue.to_csv(os.path.join('data', 'restaurant_info.csv'), index=False)
