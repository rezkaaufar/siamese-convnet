# Siamese Convolutional Neural Network for Question Answering
This work is an attempt to reproduce this paper : https://aclanthology.info/pdf/P/P16/P16-1036.pdf

Requirement :
- Python 3.6
- Tensorflow 1.4.0
- NumPy

Dataset used : https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs

- First run preprocess.py script to generate the required format for the dataset. The code is built to preprocess quora dataset.
- After the the required data is generated, you can run train.py to begin the training of the model.

Notes :
- This is not a robust implementation. There might still be some bug.

Todo:
- Create an arg parser
