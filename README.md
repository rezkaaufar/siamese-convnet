# Siamese Convolutional Neural Network for Question Answering
This work is an attempt to reproduce this paper : https://aclanthology.info/pdf/P/P16/P16-1036.pdf

Requirement :
- Python 3.6
- Tensorflow 1.4.0
- NumPy

The original paper basically aims to train a Siamese Convolutional Neural Network to predict a similarity between questions that were asked on Community Question Answering (cQA). Given an input of question pair, the model would return the similarity score and user would be presented with the most relevant questions that has already been answered.

The model was trained with Quora Question-Pair Dataset and tested with labeled Yahoo Question-Pair Dataset. You can find the training data here : https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs. For the test dataset, you have to ask the original author.

How to run the program :
- First run preprocess.py script to generate the required format for the dataset. The code is built to preprocess quora dataset.
- Then run prepare.py to generate the required format for the test dataset and the BM25 score for the final scoring. (Not that running this program require a heavy computational resource!)
- After the the required data is generated, you can run train.py to begin the training of the model.
