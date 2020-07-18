from bert_serving.client import BertClient
import itertools
import pandas as pd
import pickle
from Clustering_experiment import *
from docutils.nodes import section
from networkx.algorithms.tree import branchings
from scipy.spatial.distance import cdist
import numpy as np

from collections import defaultdict
import spacy
import textcleaner as tc
from nltk.corpus import stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))


def text_preprocessing(sentences:[str]):

    #data = list(tc.document(sentences).remove_numbers().remove_stpwrds().remove_symbols())
    lema = lemmatization(sentences=sentences)
    return pd.DataFrame(lema,columns=['Problem_Description'])


def Bert_embedding(sentences:[str]):
    bc = BertClient()
    tickets_vec = bc.encode(sentences)
    print(tickets_vec.shape)
    with open('models/BERT/Bert_representation.pickle', 'wb') as handle:
        pickle.dump(tickets_vec, handle)
    print("Embeddings Generated At models/BERT/Bert_representation.pickle")

def test_similarity(tickets_vec:np.array,topk=20):
    bc = BertClient()
    while True:
        query = input('your question: ')
        query_vec = bc.encode([query])[0]
        ##compute normalized dot product as score
        score = np.sum(query_vec * tickets_vec, axis=1) / np.linalg.norm(tickets_vec, axis=1)
        topk_idx = np.argsort(score)[::-1][:topk]
        for idx in topk_idx:
            print('> %s\t%s' % (score[idx], df[idx]))

def lemmatization(sentences:[str], allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in sentences:
        doc = nlp(sent.lower())
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out



def label_to_data(df, assigned_label):
    d = defaultdict(list)
    for i, l in enumerate(assigned_label):
        d[l].append((df[i],i))
    return d

def dict_to_df(result):
    df=pd.DataFrame(result.items(),columns=['Cluster_id', 'Tickets'])
    df.set_index(keys=['Cluster_id'], inplace=True)
    df.sort_index(inplace=True)
    return df

def load_embeddings(filepath):
    with open(file=filepath, mode='rb') as handle:
        tickets_vec = pickle.load(handle)
    return tickets_vec


df = pd.read_excel('Tickets.xlsx').values.tolist()
sentences = list(itertools.chain.from_iterable(df))
#processed_text = list(itertools.chain.from_iterable(text_preprocessing(sentences=sentences).values.tolist()))
#tickets_vec = Bert_embedding(processed_text)
tickets_vec = load_embeddings("models\BERT\Bert_representation.pickle")

kclusterer,assigned_cluster = k_means_experiment(tickets_vec,distance='cosine_distance',max_k=11)
result = label_to_data(df,assigned_cluster)
dict_to_df(result).to_csv('Output/k_means_cosine.csv')


#emclusterer= EMclusterer_experiment(means=kclusterer.means(),samples=tickets_vec)
#gaaclusterer =  Gaaclusterer_experiment(tickets_vec,k_cluster=50)
#assigned_label = label_to_data(gaaclusterer)

#spectralclusterer = SpectralClustering(affinity='nearest_neighbors').fit(tickets_vec)
#assigned_label = spectralclusterer.labels_
#result = label_to_data(df,assigned_label)
#print(result)

#affinityclusterer = AffinityPropagation().fit(tickets_vec)
#assigned_label = affinityclusterer.labels_
#l=label_to_data(df,assigned_label)

#assigned_label = agglomerative_experiment(tickets_vec,affinity='cosine')
#result=label_to_data(df,assigned_label)

#assigned_label = birch_experiement(samples=tickets_vec,n_cluster=50)
#result=label_to_data(df,assigned_label)

#clustering = DBSCAN(eps=2, min_samples=2).fit(tickets_vec)
#clustering.labels_


# from sklearn.cluster import SpectralBiclustering
# def my_func(a):
#     """Take Average a 1-D array and compare with every element if element if greater than average than 1 else 0 """
#     return [1 if i>=np.average(a) else 0 for i in a]
#
# binary_representation = np.apply_along_axis(my_func, 0, tickets_vec)
# clustering = SpectralBiclustering(n_clusters=20, random_state=0,method='log').fit(binary_representation)
# clustering.column_labels_


