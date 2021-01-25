Requirements:
stellargraph
bs4
GoogleNews-vectors-negative300.bin (in the working folder where the  .py file is)

Use '!pip install stellargraph', 'pip install bs4'

There are three files in the code folder

1. node_classification_MR.py
This file performs the document classification task as node classification task for the MR review dataset. It implements the model from Yao et. al. "Graph Convolutional Networks for Text Classification".

Run as 'python3 node_classification_MR.py' it first prints train and test accuracy for 100 epochs and then plots the tsne embeddings for GCN second layer features.

2. node_classification_Reuters.py
This file performs the document classification task as node classification task for the Reuters R8 and R52 dataset. It implements the model from Yao et. al. "Graph Convolutional Networks for Text Classification". Currently when you run the file it prints the accuracies for R8 dataset. The values for R52 dataset can be obtained by replacing 'r8-train-no-stop.txt' and 'r8-test-no-stop.txt' on lines 29 and 30 with 'r52-train-no-stop.txt' and 'r52-test-no-stop.txt' respectively.

Run as 'python3 node_classification_Reuters.py'.

3. graph_classification_MR.py
This file performs the document classification task as a graph classification task for the MR review dataset.

4. graph_classification_Reuters.py
This file performs the document classification task as a graph classification task for the Reuters R8 and R52 dataset.Currently when you run the file it prints the accuracies for R8 dataset. The values for R52 dataset can be obtained by replacing 'r8-train-no-stop.txt' and 'r8-test-no-stop.txt' on lines 30 and 31 with 'r52-train-no-stop.txt' and 'r52-test-no-stop.txt' respectively.