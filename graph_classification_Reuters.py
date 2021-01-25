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

from stellargraph.layer import GCNSupervisedGraphClassification

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt

import gensim

train_df = pd.read_csv('r8-train-no-stop.txt', sep='\t', names = ['label','text'])
test_df = pd.read_csv('r8-test-no-stop.txt', sep='\t', names = ['label','text'])
print(test_df.head())

doc_list = list(train_df['text']) + list(test_df['text'])
print(len(doc_list))

def clean_text(text):
	text = re.sub("'", "", text)
	text = re.sub("(\\W)+", " ", text)
	return text
  
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

doc_list = [clean_str(i) for i in doc_list]

doc_list = [clean_text(i) for i in doc_list]

train_labels = list(train_df['label'])
test_labels = list(test_df['label'])
labels = train_labels + test_labels
print(len(labels))

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)

dummy_y = np_utils.to_categorical(encoded_Y)


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

graphs = []

for doc in doc_list:
	d_l = doc.split(' ')
	adj_mat_source, adj_mat_destination, adj_mat_weight, feature_array = get_adj_mat_weight(doc)
	#print(len(adj_mat_source), len(adj_mat_destination), len(adj_mat_weight))

	#feature_array = np.eye(len(d_l))
	nodes = sg.IndexedArray(feature_array, index=[i for i in range(len(d_l))])
	edges = pd.DataFrame({
		"source": adj_mat_source+[i for i in range(len(d_l))],
		"target": adj_mat_destination+[i for i in range(len(d_l))],
		"weight": adj_mat_weight+[1 for i in range(len(d_l))]
	})
	# convert the raw data into StellarGraph's graph format for faster operations
	graph = sg.StellarGraph(nodes=nodes,edges=edges)
	graphs.append(graph)

generator = PaddedGraphGenerator(graphs=graphs)

def create_graph_classification_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.5,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=8, activation="softmax")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.005), loss=categorical_crossentropy, metrics=["acc"])

    return model

epochs = 200  # maximum number of training epochs
folds = 5  # the number of folds for k-fold cross validation
n_repeats = 5  # the number of repeats for repeated k-fold cross validation

es = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
)

def train_fold(model, train_gen, test_gen, es, epochs):
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=1, callbacks=[es],
    )
    # calculate performance on the test data and return along with history
    test_metrics = model.evaluate(test_gen, verbose=1)
    test_acc = test_metrics[model.metrics_names.index("acc")]

    return history, test_acc

def get_generators(train_index, test_index, graph_labels_train, graph_labels_test, batch_size):
    print(len(train_index), len(test_index))
    train_gen = generator.flow(
        train_index, targets=graph_labels_train, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels_test, batch_size=batch_size
    )
    print(len(train_gen), len(test_gen))
    return train_gen, test_gen
  
test_accs = []

for i in range(5):
	rp = np.random.permutation(7674)
	train = rp[:6139]
	test = rp[6139:]
	train_labels = dummy_y[train]
	test_labels = dummy_y[test]
	train_gen, test_gen = get_generators(
		train, test, train_labels, test_labels, batch_size=30
	)
	model = create_graph_classification_model(generator)
	history, acc = train_fold(model, train_gen, test_gen, es, epochs)
	test_accs.append(acc)