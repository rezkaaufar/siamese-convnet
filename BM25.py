import numpy as np


target = 'I have a huge dental problem ?'
answers = [['Help', 'im', 'scared', '!', 'Dental', 'problems' '?'],
['Do I need 2 dental bridges or just one huge one on my teeth?'],
['What', 'do', 'you', 'call', 'the', 'nerve', 'problem', 'involving', 'dental', 'and', 'facial', 'nerve', '?'],
['I', 'have', 'a', 'relative', 'who', 'needs', 'free', 'dental', 'work', '?'],
['Dental', 'problem', '?', 'Pulling', 'teeth', 'out', '?'],
['A', 'Deadly', 'Dental', 'Massacre', '...', '?'],
['I', 'need', 'a', 'professional', 'opinion', 'about', 'my', 'dental', 'problems', 'can', 'anyone', 'help', 'me', '?'],
['Huge', 'Dental', 'problems', '?'],
['Dental', 'problem', '.', 'help', '?' '(', 'bite', 'issue', ')', '?'],
['Dental', '-', 'Crown', 'problem', '?'],
['Problem', 'with', 'dental', 'office', 'and', 'insurance', 'predetermination', '?'],
['A', 'dental', 'problem', '??????????????????????',  '/', '...'],
['What', 'kind', 'of', 'dental', 'problems', 'cound', 'a', '69', 'year', 'old', 'male', 'have', ',', 'diagnosed', 'Diabetes', 'and', 'at', 'present', 'extremely', 'pain', '?'],
['No', 'dental', 'insurance', ',', 'but', 'a', 'huge', 'problem','.', 'Please' 'help', '.'],
['Lots', 'of', 'dental', 'pain', '(', 'nerve', 'shocks', ')', '.', 'Lots', 'of', 'insurance',',', 'but', 'no', 'money', 'for', 'the', 'huge', 'copays','.', 'What', 'to', 'do','?', 'oouch','!','?']]


def char_trigram_creator(sentence, n):
  grams = []
  words = sentence.split(" ")
  for word in words:
    word_aug = "#" + word + "#"
    for i in range(len(word_aug) - n + 1):
      grams.append(word_aug[i:i+n])
  return grams


def BM25(target, samples, k_1=1.2, b=0.75):

    target = char_trigram_creator(target, 3)
    doc_lengths = []
    n_samples = len(samples)
    counter = 0
    all_freqs = {}
    n_q = {}

    for sentence in samples:
        sentence = " ".join(sentence)
        sentence = char_trigram_creator(sentence, 3)
        doc_lengths.append(len(sentence))
        freq = {}

        for term in sentence:
            if term not in freq:
                freq[term] = 1
            else:
                freq[term] += 1

        for term in set(sentence):
            if term not in n_q:
                n_q[term] = 1
            else:
                n_q[term] += 1

        all_freqs["%d" % counter] = freq
        counter += 1

    avg_length = sum(doc_lengths)/n_samples
    idf = {k: np.log((n_samples-v+0.5)/(v+0.5) + 1) for k,v in n_q.items()}

    scores = []
    counter = 0

    for sentence in samples:

        score = 0
        sentence = " ".join(sentence)
        sentence = char_trigram_creator(sentence, 3)

        for term in sentence:
            if term in target:
                score += idf[term]*all_freqs[str(counter)][term]*(k_1+1)/(all_freqs[str(counter)][term]+k_1*(1-b+b*(sum(all_freqs[str(counter)].values())/avg_length)))
        counter += 1
        scores.append(score)

    return scores

print(BM25(target, answers))

