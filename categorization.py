from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers

import re
import ast
import string

import numpy as np
import pandas as pd
import tensorflow as tf


from collections import Counter

from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

base_url = './categorization/'

df = pd.read_csv(base_url + 'data.csv')

def text_extraction(dfi):
    sentence = ' '.join([dfi['title'], str(dfi['vendor']), dfi['tags']])
    sentence = re.sub('[^a-zA-Z0-9$.]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.lower()
    return sentence


# creating the dataset
rows = [{'text': text_extraction(df.iloc[i]), 'label': df.iloc[i]['category']} for i in range(len(df))]
dataset = pd.DataFrame(rows)

# creating integer labels for multiclass training
dataset['label_int'] = pd.Categorical(dataset['label']).codes

# extracting the names of the labels
labels_names = list(Counter(dataset['label']).keys())

max_samples = dataset['label'].value_counts().max()


balanced_data_list = []

for class_name, group in dataset.groupby('label'):
    if len(group) < max_samples:
        upsampled_group = resample(group, 
                                   replace=True, 
                                   n_samples=max_samples, 
                                   random_state=42)
    else:
        upsampled_group = group

    balanced_data_list.append(upsampled_group)

balanced_data = pd.concat(balanced_data_list)

# splitting dataset to train, validation, and test dataframes
train_df, test_df= train_test_split(balanced_data, test_size=0.2, random_state=42)
val_df = test_df.sample(frac=0.5)
test_df.drop(val_df.index, inplace=True)

# extracting texts and labels from dataframes
train_texts = train_df['text']
train_labels = train_df['label_int']
val_texts = val_df['text']
val_labels = val_df['label_int']
test_texts = test_df['text']
test_labels = test_df['label_int']

# setting the text vectorization layer with 20000 words and 320 sequence length
max_features = 20000
sequence_length = 320

vectorize_layer = layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# fitting the state of the preprocessing layer to the train set. This will cause the model to build an index of strings to integers.
vectorize_layer.adapt(train_texts)

# defining the vectorize text function
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

model = load_model(base_url + '987model')

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

def predict_category(title):
    # Step 1: Preprocess the title
    title = tf.expand_dims(title, -1)  # Expand dims to match the model's expected input
    title_vectorized = vectorize_layer(title)  # Vectorize the input title
    
    predictions = model.predict(title_vectorized)
    predicted_label = tf.argmax(predictions, axis=-1).numpy()[0]  # Get the predicted label index
    
    predicted_category = labels_names[predicted_label]
    
    return predicted_category


def predict_util(item_list):
    prediction = []
    for item in item_list:
        prediction.append(predict_category(item))

    return prediction

def list_categories_util():
    return labels_names