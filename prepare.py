import csv
import preprocess
import numpy as np
import pickle
import json
import codecs
import BM25

tri2index = preprocess.load_letter_trigram() # dictionary to convert trigram to index (for one hot vector)
TRIGRAM_SIZE = len(tri2index)

def construct_test_data():
  with open('./dataset/yahoo.data.cleaned.dat', 'r', encoding='utf8') as f:
    qa_pairs =[]
    labels = []
    test_batch = []
    for line in f:
      ar = line.split('\t')
      if len(ar) == 1:
        test_batch.append((qa_pairs,labels))
        qa_pairs = []
        labels = []
      else:
        q1_processed = preprocess.char_trigram_creator(preprocess.preprocess(ar[0]))
        q2_processed = preprocess.char_trigram_creator(preprocess.preprocess(ar[1]))
        # get the index of the array
        q1_index = [tri2index[qp] for qp in q1_processed if qp in tri2index]
        q2_index = [tri2index[ap] for ap in q2_processed if ap in tri2index]
        q1_one_hot = np.zeros((TRIGRAM_SIZE))
        q2_one_hot = np.zeros((TRIGRAM_SIZE))
        q1_one_hot[q1_index] = 1
        q2_one_hot[q2_index] = 1
        #q1_one_hot = np.asarray(q1_one_hot)
        #q2_one_hot = np.asarray(q2_one_hot)
        qa_pairs.append((q1_one_hot.tolist(), q2_one_hot.tolist()))
        labels.append(ar[2])
    xp = './dataset/test_batch.dump'
    json.dump(test_batch, codecs.open(xp, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

def calculate_BM25_score():
  with open('./dataset/yahoo.data.cleaned.dat', 'r', encoding='utf8') as f:
    question =[]
    questions = []
    scores = []
    j = 0
    for line in f:
      ar = line.split('\t')
      if len(ar) == 1:
        score = BM25.BM25(question, questions)
        scores.append(score)
        question = []
        questions = []
        j += 1
      else:
        question = ar[0]
        questions.append(ar[1])
    print(j)
    xp = './dataset/BM25_score.dump'
    json.dump(scores, codecs.open(xp, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

if __name__ == "__main__":
  construct_test_data()
  calculate_BM25_score()