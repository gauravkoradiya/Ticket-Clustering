import numpy as np
import torch
from gensim.sklearn_api import tfidf
from scipy.cluster._hierarchy import cluster_in
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
from collections import Counter
import string
import re
from CNN_LSTM_model import CNN_LSTM
from CNN_encoder import CNN_Encoder
from Early_Stopping import EarlyStopping
import itertools
from operator import itemgetter
import spacy
from sklearn.metrics.pairwise import cosine_similarity
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
from Clustering_experiment import *
from spacy.vocab import Vocab
from spacy.lang.en import English
import spacy
import os
import pandas as pd
import gensim
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix, SoftCosineSimilarity, MatrixSimilarity
from gensim.similarities import Similarity
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, SpectralClustering

word2vec_model_path = 'models/LSI_model/Word2Vec_with_new_lemma.model'
dictionary_path = 'models/LSI_model/dictionary.txt'
nlp = spacy.load('en_core_web_sm')

def new_lemmatization(sentences:[str],allowed_postags=['NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN','VBP', 'VBZ', 'JJ', 'JJR', 'JJS']):
    texts_out = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        tagged = nltk.pos_tag(tokens)
        pos_cleaned_sent = " ".join([token for (token, pos) in tagged if pos in allowed_postags])
        doc = nlp(pos_cleaned_sent)
        # Extract the lemma for each token and join
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc]))
    return texts_out

def remove_null_sentence(sentences:[str]):
    return [x for x in sentences if x is not '']


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

def text_preprocessing(sentences:[str]):

    input_text = list(tc.document(sentences).remove_numbers().remove_stpwrds().remove_symbols().lower_all())
    lema = new_lemmatization(sentences=input_text)
    return lema

def testing():
    while True:
        x=input('enter something')
        for i in cossim(x,input_sentence,dictionary):
            print(i, "------->",df[i])

df = pd.read_excel('Tickets.xlsx').values.tolist()
with open('data/preprocessed_data.pkl','rb') as handle:
    input_sentence=pickle.load(handle)
dictionary = Dictionary.load(dictionary_path)

def cossim(query, documents, dictionary,num_best=30):
    # Compute cosine similarity between the query and the documents.
    temp=[]
    temp.append(query)
    input_sentence = word_tokenizer(remove_null_sentence(text_preprocessing(sentences=temp)))
    query = dictionary.doc2bow(input_sentence[0])
    index = MatrixSimilarity([dictionary.doc2bow(document) for document in documents],num_features=len(dictionary))
    similarities = index[query]
    most_similar_index = similarities.argsort()[-num_best:][::-1]
    return most_similar_index

def generate_embeddings(documents,dictionary):
    doc_word_index = MatrixSimilarity([dictionary.doc2bow(document) for document in documents],num_features=len(dictionary))
    doc_doc_index = np.array([doc_word_index[dictionary.doc2bow(doc)] for doc in documents])
    np.save('models/Word2vec/doc_word_index',doc_word_index.index)
    np.save('models/Word2Vec/doc_doc_index',doc_doc_index)
    return doc_word_index.index,doc_doc_index

#print(doc_word_index.shape,doc_doc_index.shape)
#clusterer,assigned_cluster = k_means_experiment(sample=doc_doc_index,distance='euclidean_distance',max_k=40)

def cluster_index(data:np.array,algo:['k_means','hierarchical']='k_means',k=11,file_name='Output/k{}.csv'):
    #"l1","l2","manhattan", "cosine", or 'precomputed'.
    data_frames = []
    cluster_ind=[]
    if algo=='k_means':
        kclusterer = KMeansClusterer(num_means=k, distance=nltk.cluster.util.cosine_distance, avoid_empty_clusters=True,repeats=1)
        assigned_clusters = np.array(kclusterer.cluster(data, assign_clusters=True))
    else:
        agglomeratrive_clusterer = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage='complete').fit(data)
        assigned_clusters = np.array(agglomeratrive_clusterer.labels_)
    for i in range(k):
        cluster_index = np.where(assigned_clusters==i)[0]
        data_frames.append(pd.DataFrame(list(itertools.chain.from_iterable([df[j] for j in cluster_index]))))
        cluster_ind.append(cluster_index)
    df_keys = pd.concat(data_frames, keys=list(range(k)),names=['Cluster_no', 'Row ID']).to_csv(file_name.format(algo))
    return cluster_ind

def intra_noise_handling(cluster_index,threshold=.2):
    noisy_tickets=[]
    counter = 0
    cluster_efficiency=[]

    for i in cluster_id:
        noisy_tickets_id=[]
        a=np.array([cluster_index[j] for j in i])
        c=cosine_similarity(a,a)
        b = np.where(c<(c.mean()*threshold+c.mean()), 0, 1)
        ticket_index = np.where(b.sum(axis=0) < b.sum(axis=0).max()*threshold)
        noisy_tickets_id = [i[z] for z in ticket_index[0]]
        counter+=1
        # print("noisy ticket in cluster {} is {} ".format(counter,ticket_index[0].shape[0]))
        efficiency = b.sum()/(b.shape[0]*b.shape[1])*100
        # print("cluster efficiency : {} % ".format(efficiency))
        #
        # # for index in noisy_tickets_id:
        # #     print(index, "------>", df[index])
        # print("==================================================================")
        noisy_tickets.append(noisy_tickets_id)
        cluster_efficiency.append(efficiency)
    return noisy_tickets,cluster_efficiency

def show_cluster_information(full_cluster_index,noisy_cluster_index,cluster_efficiency):
    counter = 0
    data_frames=[]
    for cluster in zip(full_cluster_index,noisy_cluster_index,cluster_efficiency):
        print("cluster : {} with total ticket : {}".format(counter,len(cluster[0])))
        total=pd.DataFrame(list(itertools.chain.from_iterable([df[index] for index in cluster[0]]))).reset_index(drop=True)
        print("noisy tickets in cluster {} with count : {}".format(counter,len(cluster[1])))
        noisy=pd.DataFrame(list(itertools.chain.from_iterable([df[index] for index in cluster[1]]))).reset_index(drop=True)
        print("remainining ticket after removal in cluster {} is {}".format(counter,int(len(total)-len(noisy))))
        remain = pd.DataFrame(list(itertools.chain.from_iterable([df[index] for index in cluster[0] if index not in cluster[1]]))).reset_index(drop=True)
        print("cluster efficiency : {}% ".format(cluster[2]))
        print("==================================================================================================================")
        print("==================================================================================================================")
        counter+=1
        data_frames.append(pd.concat([total,noisy,remain],ignore_index=True,axis=1))
    df_info = pd.concat(data_frames, keys=list(range(len(full_cluster_index)))).to_csv('Output/cluster_info.csv')
    return df_info

#doc_word_index,doc_doc_index = generate_embeddings(documents=input_sentence,dictionary=dictionary)
doc_word_index = np.load('models/Word2vec/doc_word_index.npy')
doc_doc_index = np.load('models/Word2vec/doc_doc_index.npy')
cluster_id = cluster_index(doc_word_index,algo='k_means',k=11)
noisy_ticket_index,cluster_efficiency=intra_noise_handling(doc_word_index)
cluster_info = show_cluster_information(cluster_id,noisy_ticket_index,cluster_efficiency)
