import numpy as np
import pandas as pd
from bs4 import UnicodeDammit
from collections import defaultdict
import re
from math import log
from tqdm import tqdm

import stellargraph as sg
import tensorflow as tf

import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

path = './'

#Load and pre-process the data

with open(path+'rt-polarity.neg','rb') as file:
   content = file.read()

suggestion = UnicodeDammit(content)

encoding = suggestion.original_encoding
reviews_neg_df = pd.read_csv(path + 'rt-polarity.neg', sep = '\n', names = ['Reviews'], encoding=encoding)
print(reviews_neg_df.shape)
print(reviews_neg_df.head())

encoding = suggestion.original_encoding
reviews_pos_df = pd.read_csv(path + 'rt-polarity.pos', sep = '\n', names = ['Reviews'], encoding=encoding)
print(reviews_pos_df.shape)
print(reviews_pos_df.head())

doc_list = list(reviews_neg_df['Reviews']) + list(reviews_pos_df['Reviews'])
print(len(doc_list))

def clean_text(text):
	text = re.sub("'", "", text)
	text = re.sub("(\\W)+", " ", text)
	return text

doc_list = [clean_text(i) for i in doc_list]

pos_labels = [1 for i in range(5331)]
neg_labels = [0 for i in range(5331)]
labels = pos_labels + neg_labels
print(len(labels))

wordfreq = defaultdict(int)
for doc in doc_list:
	for word in doc.split():
		wordfreq[word]+=1
print(len(wordfreq))

for key in wordfreq.copy().keys():
	if not key.lower().islower():
		wordfreq.pop(key,None)
print(len(wordfreq))

num_docs = len(doc_list)
num_words = len(wordfreq)

word_order = {k:v for k,v in zip(list(wordfreq.keys()), range(len(list(wordfreq.keys()))))}
all_doc_vocabulary = list(wordfreq.keys())

word_frequency_in_doc = defaultdict(int)
num_docs_containing_word = defaultdict(int)

for i,doc in enumerate(doc_list):#enumerate starts from 0
	doc_word_set = set()
	for word in doc.split():
		if word not in all_doc_vocabulary:
			continue
		word_frequency_in_doc[(i,word_order[word])]+=1

		if word not in doc_word_set:
			doc_word_set.add(word)
			num_docs_containing_word[word_order[word]]+=1

print(len(num_docs_containing_word))

adj_mat_source = []
adj_mat_destination = []
adj_mat_weight = []

for doc_num in tqdm(range(num_docs)):

	for word_num in range(num_words):
		tf_idf = word_frequency_in_doc[(doc_num, word_num)]*log(1.0*num_docs/num_docs_containing_word[word_num])

		if tf_idf > 0:
			adj_mat_source.append(doc_num); adj_mat_destination.append(num_docs+word_num)
			adj_mat_source.append(num_docs+word_num); adj_mat_destination.append(doc_num)
			adj_mat_weight.append(tf_idf); adj_mat_weight.append(tf_idf)

window_size = 20
windows = []
for doc_words in doc_list:
    words = doc_words.split()
    doc_length = len(words)
    if doc_length <= window_size:
        windows.append(words)
    else:
        for i in range(doc_length - window_size + 1):
            window = words[i: i + window_size]
            windows.append(window)

# constructing all single word frequency
word_window_freq = defaultdict(int)
for window in windows:
    appeared = set()
    for word in window:
        if word not in all_doc_vocabulary:
            continue
        if word not in appeared:
            word_window_freq[word_order[word]] += 1
            appeared.add(word)


# constructing word pair count frequency
word_pair_count = defaultdict(int)
for window in tqdm(windows):
    for i in range(1, len(window)):
        for j in range(i):
            word_i = window[i]
            word_j = window[j]
            if word_i not in all_doc_vocabulary or word_j not in all_doc_vocabulary:
                continue
            word_i_id = word_order[word_i]
            word_j_id = word_order[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_count[(word_i_id, word_j_id)] += 1
            word_pair_count[(word_j_id, word_i_id)] += 1

for i in tqdm(range(num_words)):
    for j in range(num_words):
        if i == j:
            continue
        p_i = (1.0*word_window_freq[i])/len(windows)
        p_j = (1.0*word_window_freq[j])/len(windows)
        p_ij = (1.0*word_pair_count[(i,j)])/len(windows)
        if p_ij == 0:
            pmi = 0
        else:
            pmi = log(p_ij/(p_i*p_j))

        if pmi>0:
            adj_mat_source.append(num_docs+i); adj_mat_destination.append(num_docs+j)
            adj_mat_source.append(num_docs+j); adj_mat_destination.append(num_docs+i)
            adj_mat_weight.append(pmi); adj_mat_weight.append(pmi)

feature_array = np.eye(num_docs+num_words)
nodes = sg.IndexedArray(feature_array, index=[i for i in range(num_docs+num_words)])

edges = pd.DataFrame({
    "source": adj_mat_source+[i for i in range(num_docs+num_words)],
    "target": adj_mat_destination+[i for i in range(num_docs+num_words)],
    "weight": adj_mat_weight+[1 for i in range(num_docs+num_words)],
})

# convert the raw data into StellarGraph's graph format for faster operations
graph = sg.StellarGraph(nodes=nodes,edges=edges)
print(graph.info())

my_subs = pd.Series(['apple' for i in range(5331)]+['bannana' for i in range(5331)]).rename('subject') 

train_subjects, test_subjects = model_selection.train_test_split(
    my_subs, train_size=7464, test_size=None, stratify=my_subs, random_state = 0,
)
val_subjects, test_subjects = model_selection.train_test_split(
    test_subjects, train_size=1066, test_size=None, stratify=test_subjects
)

target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects)
train_targets = np.hstack((train_targets, 1 - train_targets))

val_targets = target_encoding.transform(val_subjects)
val_targets = np.hstack((val_targets, 1 - val_targets))

test_targets = target_encoding.transform(test_subjects)
test_targets = np.hstack((test_targets, 1 - test_targets))

generator = sg.mapper.FullBatchNodeGenerator(graph, method="gcn")
train_gen = generator.flow(train_subjects.index, train_targets)

# two layers of GCN, each with hidden dimension 16
gcn = sg.layer.GCN(layer_sizes=[200, 16], activations=["relu", "relu"], generator=generator, dropout=0.5)
x_inp, x_out = gcn.in_out_tensors() # create the input and output TensorFlow tensors

predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

model = Model(inputs=x_inp, outputs=predictions)
model.compile(
    optimizer=optimizers.Adam(lr=0.01),
    loss=losses.categorical_crossentropy,
    metrics=["acc"],
)

val_gen = generator.flow(val_subjects.index, val_targets)

es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)

history = model.fit(
    train_gen,
    epochs=100,
    validation_data=val_gen,
    verbose=2,
    shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
    callbacks=[es_callback],
)

all_nodes = my_subs.index
all_gen = generator.flow(all_nodes)
all_predictions = model.predict(all_gen)

embedding_model = Model(inputs=x_inp, outputs=x_out)

emb = embedding_model.predict(all_gen)

transform = TSNE

X = emb.squeeze(0)

trans = transform(n_components=2)
X_reduced = trans.fit_transform(X)
X_reduced.shape

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=my_subs.astype("category").cat.codes,
    cmap="jet",
    alpha=0.7,
)
ax.set(
    aspect="equal",
    xlabel="$X_1$",
    ylabel="$X_2$",
    title=f"{transform.__name__} visualization of GCN embeddings for MR dataset",
)