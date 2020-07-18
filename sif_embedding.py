from sklearn.ensemble._gradient_boosting import np_bool
from src import data_io,params,SIF_embedding
import pickle
import textcleaner as tc
import spacy
import math
import numpy as np
import torch
from sklearn.metrics import davies_bouldin_score,silhouette_score,calinski_harabaz_score
import matplotlib.pyplot as plt
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
from jqmcvi import base
import pandas as pd
import gensim
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix, SoftCosineSimilarity, MatrixSimilarity
from gensim.similarities import Similarity
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, SpectralClustering
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import os

nlp = spacy.load('en_core_web_sm')

# input Hyper parameter
Word2Vec_path = 'models/Word2vec'
SIF_embedding_path= 'models/SFI'
Cluster_ouput_path = 'Output'

word2vecfile = 'model.txt' # word vector file, can be downloaded from GloVe website
weightfile = 'freqency_dict.txt' # each line is a word and its frequency
sif_file='SIF_embeddings'
cluster_output_file = 'cluster_{}_{}.csv'
cluster_info_file = 'cluster_info.csv'
pure_cluster_file = 'pure_cluster_{}.csv'
noisy_cluster_file = 'noisy_cluster_{}.csv'

weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme
Desired_cluster = 25
Min_cluster = 2
Binary_threshold = 0.80
Noise_threshold = 0.4
Fixed_K = 15
Power_constant = 0.5
Window_size_for_k = 5
Pure_cluster_threshold = 0.8
Cluster_size = 3

algo='hierarchical'#['k_means','hierarchical']
cluster_scoring_function = 'davies_bouldin_score'#['silhouette_score','davies_bouldin_score','calinski_harabaz_score','dunn']

SIF_embedding_generate = False
pretrained_word2vec = True
test = True
k_calculation = False

def new_lemmatization(sentences:[str],allowed_postags=['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN','VBP', 'VBZ', 'JJ', 'JJR', 'JJS']):#'RB', 'RBR', 'RBS',
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



def load_dict_from_file(path):
    frequency_dict = {}
    with open(os.path.join(path,weightfile), 'r') as raw_data:
        for item in raw_data:
            key,value = item.rstrip().split(' ',1)
            frequency_dict[key] = int(value)
    return frequency_dict


def save_dict_to_file(word2vec_model,path):
    with open(os.path.join(path,weightfile), 'w') as f:
        f.writelines('{} {}\n'.format(word, freq.count) for word, freq in word2vec_model.wv.vocab.items())
    print('== Dictionary Saved at {} ==='.format(os.path.join(path,weightfile)))
    return os.path.join(path,weightfile)


def train_word2vec(Input_sentence, Embedding_size,path):
    model = Word2Vec(Input_sentence, size=Embedding_size, min_count=1, workers=8,compute_loss=True) #replace with bigram_sent,sentence for bigram model, unigram evaluation
    print(model)
    model.train(Input_sentence,total_examples=len(Input_sentence),epochs=10)
    #model.init_sims(replace=True)
    model.save(os.path.join(path,'Word2Vec.model'))
    model.wv.save_word2vec_format(os.path.join(path,'W2V_model.txt'),binary=False)
    print("model is trained and saved at {}".format(os.path.join(path,'W2V_model.txt')))
    return model

def generate_SFI_embedding(sentences,words,We,weight4ind,params,path,phase='training'):
    # load sentences
    x, m, = data_io.sentences2idx(sentences, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    w = data_io.seq2weight(x, m, weight4ind) # get word weights
    # get SIF embedding
    total_embedding = SIF_embedding.SIF_embedding(We, x, w, params,phase=phase) # embedding[i,:] is the embedding for sentence i
    np.save(os.path.join(path,sif_file),total_embedding)
    print("==========embeddings saved at {} ========".format(os.path.join(path,sif_file)))
    return total_embedding

def calculate_cosine_similarity(a,b,df,k=30):
    #print("Actual ticket : ",df[index])
    similarity_matrix = cosine_similarity(a.reshape(1,-1), b)
    index = similarity_matrix[0].argsort()[-k:][::-1]
    for i in index:
        print(i, "------->", df.iloc[i].values,"---->",similarity_matrix[0][i])
    return index

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

def testing_embedding(total_embedding,words,weight4ind,We,df):

    temp=[]
    x=input("Enter ticket index : ")
    temp.append(x)
    preprocessed_sentence = remove_null_sentence(text_preprocessing(sentences=temp))
    # load sentences
    x, m, = data_io.sentences2idx(preprocessed_sentence,words)  # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    w = data_io.seq2weight(x, m, weight4ind)  # get word weights
    # get SIF embedding
    input_embedding = SIF_embedding.SIF_embedding(We, x, w, params,phase='testing')  # embedding[i,:] is the embedding for sentence i
    index = calculate_cosine_similarity(input_embedding,total_embedding,df)

def clustering_engine(data:np.array,df,file_name,algo:['k_means','hierarchical']='k_means',k=11):
    #affinity in Agglomeratigve could be "l1","l2","manhattan", "cosine", or 'precomputed'.
    data_frames = []
    cluster_ind=[]
    if algo == 'k_means':
        clusterer = KMeansClusterer(num_means=k, distance=nltk.cluster.util.cosine_distance, avoid_empty_clusters=True,repeats=1,normalise=True)
        assigned_clusters = np.array(clusterer.cluster(data, assign_clusters=True))
    else:
        clusterer = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage='complete').fit(data)
        assigned_clusters = np.array(clusterer.labels_)

    for i in range(k):
        cluster_index = np.where(assigned_clusters==i)[0]
        data_frames.append(df.iloc[cluster_index])
        #data_frames.append(pd.DataFrame(list(itertools.chain.from_iterable([df.iloc[j].values[0] for j in cluster_index]))))
        cluster_ind.append(cluster_index)
    #create a CSV file
    clust_info = pd.concat(data_frames, keys=list(range(k)),names=['Cluster_no', 'Row ID']).to_csv(os.path.join(Cluster_ouput_path,file_name.format(k,algo)))
    return clusterer,cluster_ind,assigned_clusters

def cluster_overlapping_score(sentence_embeddings,cluster_id,churn_threshold = 0.5):

    assigned_cluster_id=cluster_id
    total_mean_info= []
    for one_cluster_index in assigned_cluster_id:
        cluster_mean=[]
        a = np.array([sentence_embeddings[j] for j in one_cluster_index])
        print(a.shape)
        for each_cluster in assigned_cluster_id:
            b = np.array([sentence_embeddings[j] for j in each_cluster])
            print(cosine_similarity(a,b).shape)
            cluster_mean.append(cosine_similarity(a,b).mean())
        total_mean_info.append(cluster_mean)

def inter_noise_handling(sentence_embeddings,cluster_id,binary_threshold=.2,noise_threshold=0.5):

    efficiency=[]
    for i in cluster_id:
        one_cluster_index=np.array([sentence_embeddings[j] for j in i])
        similarity_matrix=cosine_similarity(one_cluster_index,one_cluster_index)
        binay_matrix = np.where(similarity_matrix < binary_threshold, 0, 1)# np.where(similarity_matrix<(similarity_matrix.mean()+(similarity_matrix.mean()*binary_threshold)), 0, 1)
        efficiency.append((binay_matrix.sum(axis=0).mean())/len(binay_matrix)*100)
    return efficiency

def clustering_function(data:np.array,df,algo:['k_means','hierarchical']='k_means',k=15):

    cluster_ind = []
    if algo == 'k_means':
        clusterer = KMeansClusterer(num_means=k, distance=nltk.cluster.util.cosine_distance, avoid_empty_clusters=True, repeats=1, normalise=True)
        assigned_clusters = np.array(clusterer.cluster(data, assign_clusters=True))
    else:
        clusterer = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage='complete').fit(data)
        assigned_clusters = np.array(clusterer.labels_)

    for i in range(k):

        cluster_index = np.where(assigned_clusters==i)[0]
        cluster_ind.append(cluster_index)

    return cluster_ind

def noise_handling(sentence_embeddings, df, cluster_id, binary_threshold=.2, noise_threshold=0.5):

    noisy_tickets=[]
    pure_cluster_set = []

    for i in cluster_id:
        if len(i) < Cluster_size :
            noisy_tickets.append(i)
        else:
            one_cluster_index=sentence_embeddings[i]
            similarity_matrix=cosine_similarity(one_cluster_index,one_cluster_index)
            binary_matrix = np.where(similarity_matrix < binary_threshold, 0, 1)#np.where(similarity_matrix<(similarity_matrix.mean()+(similarity_matrix.mean()*binary_threshold)), 0, 1)

            noisy_ticket_index = np.where(binary_matrix.sum(axis=0) <= math.ceil(noise_threshold*len(binary_matrix)))[0]#b.sum(axis=0).max()*threshold)
            noisy_tickets_id = np.take(i,noisy_ticket_index) #or use[i[z] for z in noisy_ticket_index[0]]

            remaining_tickets_id = np.delete(i,noisy_ticket_index)
            remaining_index = np.array([np.argwhere(i == j) for j in remaining_tickets_id]).reshape(-1)

            # calculate a cluster efficiency with noise
            with_noise_efficiency_score = (binary_matrix.sum(axis=0).mean()) / len(binary_matrix) * 100

            if len(noisy_tickets_id) > math.floor(len(i)*0.8):

                if len(remaining_index)> Cluster_size:
                    # calculate a cluster efficiency without noise
                    binary_matrix_without_noise = np.delete(binary_matrix[remaining_index], noisy_ticket_index, axis=1)
                    without_noise_binary_matrix_score = (binary_matrix_without_noise.sum(axis=0).mean()) / len(binary_matrix_without_noise) * 100

                    if without_noise_binary_matrix_score > Pure_cluster_threshold * 100:
                        pure_cluster_set.append((remaining_tickets_id,without_noise_binary_matrix_score))
                    else:
                        noisy_tickets.append((remaining_tickets_id,without_noise_binary_matrix_score))

                noisy_tickets.append((noisy_tickets_id,with_noise_efficiency_score))
            else:
                # calculate a cluster efficiency without noise
                binary_matrix_without_noise= np.delete(binary_matrix[remaining_index],noisy_ticket_index,axis=1)
                without_noise_binary_matrix_score = (binary_matrix_without_noise.sum(axis=0).mean()) / len(binary_matrix_without_noise) * 100

                noisy_tickets.append((noisy_tickets_id,with_noise_efficiency_score))

                if without_noise_binary_matrix_score > Pure_cluster_threshold*100:
                    pure_cluster_set.append((remaining_tickets_id,without_noise_binary_matrix_score))
                else:
                    noisy_tickets.append((remaining_tickets_id,without_noise_binary_matrix_score))

    pure_cluster = save_cluster(cluster_set=pure_cluster_set,df=df,path=Cluster_ouput_path,label='pure')
    noisy_cluster = save_cluster(cluster_set=noisy_tickets,df=df,path=Cluster_ouput_path,label='noisy')
    #cluster_info = show_cluster_information(full_cluster_index=cluster_id, noisy_cluster_index=noisy_tickets, df=df, path=Cluster_ouput_path)
    #if Recursion_phase > 1:
    # if len(pure_cluster) != 0:
    #     # Do Cluster again on Noisy Tickets
    #     sentence_embeddings = sentence_embeddings[list(itertools.chain.from_iterable(noisy_tickets))]
    #     df = df.iloc[list(itertools.chain.from_iterable(noisy_tickets))]#[df[index] for index in list(itertools.chain.from_iterable(noisy_tickets))]
    #     k = math.ceil(pow(len(sentence_embeddings), Power_constant))
    #
    #     _,assigned_cluster_id,_ = clustering_engine(data=sentence_embeddings,df=df,file_name=cluster_output_file,algo=algo,k=k)
    #     #assigned_cluster_id = clustering_function(data=sentence_embeddings, algo=algo, k=k)
    #
    #     result.append(noise_handling(sentence_embeddings=sentence_embeddings, df=df, cluster_id=assigned_cluster_id, binary_threshold=binary_threshold,noise_threshold=noise_threshold))

    return pure_cluster,noisy_cluster

def save_cluster(cluster_set,df,path,label:['noisy','pure']='pure'):
    counter = 0
    data_frames = []
    for one_cluster in cluster_set:
        cluster = one_cluster[0]
        score = one_cluster[1]
        cluster_df = df.iloc[cluster]
        cluster_df.columns = ['cluster {}'.format(counter)]
        cluster_df = cluster_df.append({cluster_df.columns.values[0]: 'Score = {}'.format(score)},ignore_index=True)
        data_frames.append(cluster_df)
        counter=counter+1
    if len(data_frames)>1:
        cluster_df = pd.concat(objs=data_frames, axis=1)
    else:
        if data_frames :
            cluster_df = data_frames[0]
        else:
            return pd.DataFrame()
    cluster_df.to_csv(os.path.join(path,label+" cluster_{}.csv".format(counter)))
    return cluster_df


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_k(sentence_embeddings,df,algo,cluster_scoring_function):

    score=[]
    Max_k = math.ceil(pow(len(sentence_embeddings),Power_constant))
    for k in range(Min_cluster,Max_k):
       _,assigned_cluster_id,assigned_cluster = clustering_engine(data=sentence_embeddings,df=df,file_name=cluster_output_file,algo=algo,k=k)
       if cluster_scoring_function == 'davies_bouldin_score':
           score.append(davies_bouldin_score(sentence_embeddings, assigned_cluster))
       elif cluster_scoring_function == 'silhouette_score':
           score.append(silhouette_score(sentence_embeddings, assigned_cluster))
       elif cluster_scoring_function == 'calinski_harabaz_score':
           score.append(calinski_harabaz_score(sentence_embeddings, assigned_cluster))
       else:
           score.append(base.dunn_fast(sentence_embeddings,assigned_cluster))
    #plot_cluster_score(score=score,Min_cluster=Min_cluster,Max_cluster=int(len(df)/Desired_cluster),algo=algo,scoring_function=cluster_scoring_function)
    k_index = np.argsort(score)[1:Window_size_for_k]
    final_k = find_nearest(k_index,Max_k)
    return final_k


def main():

    df = pd.read_excel('Tickets.xlsx')
    sentences = df.values.flatten().tolist()
    preprocessed_sentences = remove_null_sentence(text_preprocessing(sentences=sentences))
    input_sentence = word_tokenizer(preprocessed_sentences)
    # sentences= []
    # with open('data/preprocessed_data.pkl', 'rb') as handle:
    #     input_sentences = pickle.load(handle)
    # for i in input_sentences:
    #     sentences.append(' '.join(i))

    if pretrained_word2vec :
        # load word vectors
        words, We = data_io.getWordmap(textfile=os.path.join(Word2Vec_path,word2vecfile))
        # load word weights
        word2weight = data_io.getWordWeight(os.path.join(Word2Vec_path,weightfile),a=weightpara)  # word2weight['str'] is the weight for the word 'str'
        weight4ind = data_io.getWeight(words, word2weight)  # weight4ind[i] is the weight for the i-th word
    else:
        word2vec_model = train_word2vec(Input_sentence=input_sentence, Embedding_size=100, path=Word2Vec_path)
        weightfile_path = save_dict_to_file(word2vec_model=word2vec_model,path=Word2Vec_path)
        # load word vectors
        words, We = data_io.getWordmap(textfile=os.path.join(Word2Vec_path,word2vecfile))
        # load word weights
        word2weight = data_io.getWordWeight(weightfile=os.path.join(Word2Vec_path,weightfile),a=weightpara)  # word2weight['str'] is the weight for the word 'str'
        weight4ind = data_io.getWeight(words, word2weight)  # weight4ind[i] is the weight for the i-th word
        #load_dict_from_file(path='models/Word2vec/freqency_dict.txt')

    if SIF_embedding_generate:
        sentence_embeddings = generate_SFI_embedding(sentences=preprocessed_sentences,words=words,We=We,weight4ind=weight4ind,params=params,path=SIF_embedding_path,phase='training')
    else:
        sentence_embeddings = np.load(os.path.join(SIF_embedding_path,'SIF_embeddings.npy'))

    if test:
        testing_embedding(total_embedding=sentence_embeddings,words=words,weight4ind=weight4ind,We=We,df=df)

    if k_calculation:
        k_range = find_k(sentence_embeddings=sentence_embeddings,df=df,algo=algo,cluster_scoring_function=cluster_scoring_function)
    else:
        k_range = Fixed_K

    _, assigned_cluster_id, assigned_cluster = clustering_engine(data=sentence_embeddings, df=df, file_name=cluster_output_file, algo=algo, k=k_range)
    noisy_ticket_index, cluster_efficiency, remaining_ticket_index = noise_handling(sentence_embeddings,cluster_id=assigned_cluster_id,df=df,binary_threshold=Binary_threshold,noise_threshold=Noise_threshold)
    #cluster_info = show_cluster_information(assigned_cluster_id, noisy_ticket_index,df=df,path=Cluster_ouput_path)


if __name__ == "__main__":
    # set parameters
    params = params.params()
    params.rmpc = rmpc
    main()