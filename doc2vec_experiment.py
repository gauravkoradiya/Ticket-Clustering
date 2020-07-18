import numpy as np
import torch
from gensim.sklearn_api import tfidf
from sklearn.cluster import Birch
from sklearn.metrics import davies_bouldin_score
from torch import *
import torch.nn as nn
import torch.optim as optim
from pygments.lexer import words
from setuptools.command.dist_info import dist_info
from spacy.syntax.nn_parser import precompute_hiddens
from torch.autograd import Variable
import pandas as pd
import random
from torch.utils.data import DataLoader
import itertools
import nltk
from nltk.corpus import stopwords
import random
from Clustering_experiment import *
from collections import Counter
import string
from sklearn import metrics
import matplotlib.pyplot as plt

import re
from CNN_LSTM_model import CNN_LSTM
from CNN_encoder import CNN_Encoder
from Early_Stopping import EarlyStopping
import itertools
from operator import itemgetter
import spacy
import pandas as pd
from gensim.corpora import Dictionary
import textcleaner as tc
from docutils.nodes import section, sidebar
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize,word_tokenize
import string
import torch.nn.functional as F
import pickle
from spacy.language import Language
import EncoderRNN as en
import DecoderRNN as dc
from torch import *
from spacy.vocab import Vocab
from spacy.lang.en import English
import spacy
import os
import pandas as pd
import gensim
from gensim.corpora import Dictionary
from gensim.models.doc2vec import Doc2Vec, TaggedDocument,LabeledSentence
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix, SoftCosineSimilarity, MatrixSimilarity
from gensim.similarities import Similarity

word2vec_model_path = 'models/LSI_model/Word2Vec_with_new_lemma.model'
dictionary_path = 'models/LSI_model/dictionary.txt'

nlp = spacy.load('en_core_web_sm')

def word_tokenizer(sentences:[str]):

    sentence=[]
    for raw_sentence in sentences:
    # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
        # Otherwise, get a list of words
            sentence.append(word_tokenize(raw_sentence))
    return sentence
    # Return the list of sentences
    # so this returns a list of lists
#Load Dataset
df = pd.read_excel('Tickets.xlsx').values.tolist()


with open('data/preprocessed_data.pkl','rb') as handle:
    input_sentence=pickle.load(handle)

dictionary = Dictionary.load(dictionary_path)
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(input_sentence)]
# model = Doc2Vec(documents, vector_size=200, window=3, min_count=2)
model = Doc2Vec(vector_size=50, min_count=1, epochs=40)
model.build_vocab(documents)
model.train(documents,total_examples=model.corpus_count,epochs=model.epochs)
# # store the model to mmap-able files
#model.save('models/Word2vec/model.doc2vec')
# load the model back
#model = Doc2Vec.load('models/Word2vec/model.doc2vec')

print(model.most_similar_cosmul(input_sentence[1]))
# temp=[]
# for i in input_sentence:
#     temp.append(model.infer_vector(i))
# representation = np.stack(temp,axis=0)
# np.save('doc2vec_representation',representation)
# representation = np.load('doc2vec_representation.npy')
#
# assigned_cluster = agglomerative_experiment(representation)

from sklearn.metrics.pairwise import cosine_similarity
# while True:
#     temp=[]
#     query = input("Enter something : ")
#     temp.append(query)
#     preprocessed_sentences = remove_null_sentence(text_preprocessing(sentences=temp))
#     input_sentence = word_tokenizer(preprocessed_sentences)
#     print(model.most_similar_cosmul(input_sentence[0]))
#     rp = model.infer_vector(input_sentence[0])
#     similarity=cosine_similarity(rp.reshape(1,-1),representation)
#     index = similarity[0].argsort()[-30:][::-1]
#     for i in index:
#         print(i, "------->", df[i])
# Pick a random document from the test corpus and infer a vector from the model
# doc_id = random.randint(0, len(documents) - 1)
# inferred_vector = model.infer_vector(documents[doc_id].words)
# sims = model.docvecs.most_similar([inferred_vector],topn=len(model.docvecs))
#
# # Compare and print the most/median/least similar documents from the train corpus
# print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(documents[doc_id].words)))
# print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
# for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
#     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(documents[sims[index][0]].words)))