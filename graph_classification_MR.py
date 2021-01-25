import numpy as np
import pandas as pd
from bs4 import UnicodeDammit
from collections import defaultdict
import re
from math import log
from tqdm import tqdm

import pandas as pd
import numpy as np

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph

from stellargraph import datasets

from sklearn import model_selection
from IPython.display import display, HTML

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt

import gensim

path = './'

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

def remove_numbers(text):
	removed_numbers = re.sub(r'\d', '', text)
	return removed_numbers

doc_list = [clean_text(i) for i in doc_list]

doc_list_pos = list(reviews_neg_df['Reviews'])
doc_list_neg = list(reviews_pos_df['Reviews'])

doc_list_pos = [clean_text(i) for i in doc_list_pos]
doc_list_neg = [clean_text(i) for i in doc_list_neg]

doc_list_pos = [remove_numbers(i) for i in doc_list_pos]
doc_list_neg = [remove_numbers(i) for i in doc_list_neg]

embedding_path = "GoogleNews-vectors-negative300.bin"
word2vec_google_news_300 = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=True)

def get_adj_mat_weight(doc):
	feature_array = []
	adj_mat_source = [] 
	adj_mat_destination = [] 
	adj_mat_weight = []

	d_l = doc.split(' ')
	feature_array = []
	for i in range(len(d_l)):
		try:
			feature_array.append(np.array(word2vec_google_news_300[d_l[i]]))
		except:
			feature_array.append(np.array([0]*300))
	
	for i in range(len(d_l)):
		for j in range(i+1, len(d_l)):
			adj_mat_source.append(i); adj_mat_source.append(j)
			adj_mat_destination.append(j); adj_mat_destination.append(i)

			try:
				similarity_score = word2vec_google_news_300.similarity(d_l[i], d_l[j])
			except:
				similarity_score = 0.000001
			adj_mat_weight.append(similarity_score); adj_mat_weight.append(similarity_score)
	return adj_mat_source, adj_mat_destination, adj_mat_weight, np.array(feature_array)

graphs_pos = []
graphs_neg = []

for doc in doc_list_pos:
	d_l = doc.split(' ')
	if len(d_l)>=10:
		adj_mat_source, adj_mat_destination, adj_mat_weight, feature_array = get_adj_mat_weight(doc)
		print(len(adj_mat_source), len(adj_mat_destination), len(adj_mat_weight))

		#feature_array = np.eye(len(d_l))
		nodes = sg.IndexedArray(feature_array, index=[i for i in range(len(d_l))])
		edges = pd.DataFrame({
			"source": adj_mat_source+[i for i in range(len(d_l))],
			"target": adj_mat_destination+[i for i in range(len(d_l))],
			"weight": adj_mat_weight+[1 for i in range(len(d_l))]
		})
		# convert the raw data into StellarGraph's graph format for faster operations
		graph = sg.StellarGraph(nodes=nodes,edges=edges)
		graphs_pos.append(graph)

for doc in doc_list_neg:
	d_l = doc.split(' ')
	if len(d_l)>=10:
		adj_mat_source, adj_mat_destination, adj_mat_weight, feature_array = get_adj_mat_weight(doc)
		print(len(adj_mat_source), len(adj_mat_destination), len(adj_mat_weight))

		#feature_array = np.eye(len(d_l))
		nodes = sg.IndexedArray(feature_array, index=[i for i in range(len(d_l))])
		edges = pd.DataFrame({
			"source": adj_mat_source+[i for i in range(len(d_l))],
			"target": adj_mat_destination+[i for i in range(len(d_l))],
			"weight": adj_mat_weight+[1 for i in range(len(d_l))]
		})

		# convert the raw data into StellarGraph's graph format for faster operations
		graph = sg.StellarGraph(nodes=nodes,edges=edges)
		graphs_neg.append(graph)

print(len(graphs_pos), len(graphs_neg))
graphs = graphs_pos + graphs_neg
graph_labels = [0 for i in range(len(graphs_pos))] + [1 for i in range(len(graphs_neg))] 
graph_labels = pd.Series(graph_labels, name = 'label',  dtype="category")

summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
    columns=["nodes", "edges"],
)
print(summary.describe().round(1))

graph_labels = pd.get_dummies(graph_labels, drop_first=True)
generator = PaddedGraphGenerator(graphs=graphs)

k = 35  # the number of rows for the output tensor
layer_sizes = [128, 64, 32, 1]

dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    k=k,
    bias=False,
    generator=generator,
)
x_inp, x_out = dgcnn_model.in_out_tensors()

x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)

x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=128, activation="relu")(x_out)
x_out = Dropout(rate=0.5)(x_out)

predictions = Dense(units=1, activation="sigmoid")(x_out)

model = Model(inputs=x_inp, outputs=predictions)

model.compile(
    optimizer=Adam(lr=0.0001), loss=binary_crossentropy, metrics=["acc"],
)

train_graphs, test_graphs = model_selection.train_test_split(
    graph_labels, train_size=0.9, test_size=None, stratify=graph_labels,
)

gen = PaddedGraphGenerator(graphs=graphs)

train_gen = gen.flow(
    list(train_graphs.index - 1),
    targets=train_graphs.values,
    batch_size=50,
    symmetric_normalization=False,
)

test_gen = gen.flow(
    list(test_graphs.index - 1),
    targets=test_graphs.values,
    batch_size=1,
    symmetric_normalization=False,
)

epochs = 100
history = model.fit(
    train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=True,
)