# pre-processed by lower-casing, stemming, stopword and special character removal

from nltk.stem.porter import *
from nltk.corpus import stopwords # needs to download corpus from nltk
import re
import pickle
import csv
import xml.etree.ElementTree as etree
import json
import numpy as np
import codecs
import itertools
from string import ascii_lowercase

DATA_PATH = "./dataset/quora_duplicate_questions.tsv"

def _lowercased(sentence):
  return sentence.lower()

# only remove punctuation
def _special_char_removal(sentence):
  #regex = re.compile('[^a-z]')
  #replaced = regex.sub(' ',sentence) # replace non alphabet from sentence
  #return " ".join(replaced.split()) # remove duplicated space
  #return re.sub(r'[^\w\s]', '', sentence)
  return re.sub(r'[^A-Za-z0-9]+', '', sentence)

def _stemmer(sentence):
  stemmer = PorterStemmer()
  words = sentence.split()
  stemmed_words = []
  for word in words:
    stemmed_words.append(stemmer.stem(word))
  return " ".join(stemmed_words)

def _stop_word_removal(sentence):
  words = sentence.split()
  filtered_words = [word for word in words if word not in stopwords.words('english')]
  return " ".join(filtered_words)

def preprocess(sentence):
  lowercased = _lowercased(sentence)
  sc_removed = _special_char_removal(lowercased)
  sw_removed = _stop_word_removal(sc_removed)
  stemmed = _stemmer(sw_removed)
  return stemmed

def char_trigram_creator(sentence):
  n = 3
  grams = []
  words = sentence.split(" ")
  for word in words:
    word_aug = "#" + word + "#"
    for i in range(len(word_aug) - n + 1):
      grams.append(word_aug[i:i+n])
  return grams

def save_letter_trigram():
  """
    Calculate how many letter trigram exists in the dataset
    Args:
  """
  letter_trigram_set = set()
  tri2index = {}
  i = 0
  with open(DATA_PATH) as f:
    lines = csv.reader(f, delimiter="\t")
    for line in lines:
      q_processed = char_trigram_creator(preprocess(line[3]))
      a_processed = char_trigram_creator(preprocess(line[4]))
      # get the index of the array
      for qp in q_processed:
        letter_trigram_set.add(qp)
      for ap in a_processed:
        letter_trigram_set.add(ap)
      if i % 100000 == 0 and i != 0: print(i)
      i += 1
  lt = sorted(list(letter_trigram_set))
  for i, elem in enumerate(lt):
    tri2index[elem] = i
  pickle.dump(tri2index, open('./dataset/tr2index.dump', 'wb'))

def load_letter_trigram(path = './dataset/tr2index.dump'):
  return pickle.load(open(path, 'rb'))

def print_data(i):
  """
    Prints the data
    Args:
      i: How many Question and Answer to print
  """
  with open(DATA_PATH) as f:
    lines = csv.reader(f, delimiter="\t")
    i = 0
    for line in lines:
      print("original question : {}".format(line[3]))
      print("retrieved trigram : {}".format(char_trigram_creator(preprocess(line[3]).replace(" ", ""))))
      i += 1
      if i == 10: break
    f.close()

def construct_dataset(tri2index, num_examples=100, qap_path="./dataset/qa_pairs.dump",
                      label_path="./dataset/labels.dump"):
  """
    Construct dataset to have a tuples of list for question and answer, and an integer for label
    and convert it into one hot vector
    Args:
      i: How many number of examples to load
  """
  Xpair = []
  Y = []
  with open(DATA_PATH) as f:
    lines = csv.reader(f, delimiter="\t")
    i = 0
    for line in lines:
      q_processed = char_trigram_creator(preprocess(line[3]))
      a_processed = char_trigram_creator(preprocess(line[4]))
      # get the index of the array
      q_index = [tri2index[qp] for qp in q_processed]
      a_index = [tri2index[ap] for ap in a_processed]
      # append dataset
      Xpair.append((q_index, a_index))  # list of tuples
      Y.append(line[5].replace("\n", ""))
      # if i >= num_examples : break
      i += 1
    f.close()
  # shuffle the dataset
  Xs = np.asarray(Xpair[1:])  # discard the header
  Ys = np.asarray(Y[1:])
  perm = np.random.permutation(len(Xs))
  Xs = Xs[perm]
  Ys = Ys[perm]
  pickle.dump(Xs, open(qap_path, "wb"))
  pickle.dump(Ys, open(label_path, "wb"))

if __name__ == "__main__":
  save_letter_trigram()
  tri2index = load_letter_trigram() # dictionary to convert trigram to index (for one hot vector)
  construct_dataset(tri2index)