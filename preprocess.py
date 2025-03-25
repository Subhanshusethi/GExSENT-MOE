import pandas as pd
import pickle
import random

# Load the data

df1 = pd.read_csv('df_MOSEI_combined_train.tsv', sep='\t')

with open('FACET_aligned_MOSEI.pkl', 'rb') as f:
    data = pickle.load(f)

# Ensure 'frames' column is treated as a string

df1['frames'] = df1['frames'].astype(str).str.replace('--', '').str.replace('-', '')

data_list = []

# Extract relevant data
for key in data.keys():
    if key in df1['frames'].values:
        print(f'Processing key: {key}')

        try:
            features = data[key]['features']
        except KeyError:
            print(f'Key {key} not found in data.')
            continue

        row = df1[df1['frames'] == key]
        sentences = row['sentence'].values
        polarity = row['polarity_combined'].values

        data_list.append({
            'key': key,
            'features': features,
            'sentences': sentences,
            'polarity': polarity
        })

# Shuffle data before splitting
random.shuffle(data_list)

# 80-20 split
split_index = int(0.8 * len(data_list))
train_data = data_list[:split_index]
eval_data = data_list[split_index:]

# Save training data
with open('processed_train_data.pkl', 'wb') as f_train:
    pickle.dump(train_data, f_train)

# Save evaluation data
with open('processed_eval_data.pkl', 'wb') as f_eval:
    pickle.dump(eval_data, f_eval)

print('Data saved successfully in "processed_train_data.pkl" and "processed_eval_data.pkl".')   