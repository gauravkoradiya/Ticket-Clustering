import collections
import itertools

import nltk
import spacy
import pandas as pd
import textcleaner as tc
from docutils.nodes import section
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize,word_tokenize
import gensim
import pandas as pd
import string
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from random import choice
nlp = spacy.load('en_core_web_sm')

def sentence_tokenizer(sentences:str):
    raw_sentences = []
    doc = nlp(sentences)
    for i, token in enumerate(doc.sents):
        print('-->Sentence %d: %s' % (i, token.text))
        raw_sentences.append(token.text)
    return raw_sentences

def text_preprocessing(sentences:[str]):

    input_text = list(tc.document(sentences).remove_numbers().remove_stpwrds().remove_symbols().lower_all())
    lema = new_lemmatization(sentences=input_text)
    return lema

# def lemmatization(sentences:[str], allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
#     """https://spacy.io/api/annotation"""
#     texts_out = []
#     for sent in sentences:
#         doc = nlp(sent.lower())
#         texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
#     return texts_out

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

def word_count_per_document(input_sentences):
    dictionary = Dictionary(input_sentences)

    return pd.DataFrame.from_dict({dictionary[id]:dictionary.dfs[id] for id in dictionary.dfs},orient='index')

df = pd.read_excel('Tickets.xlsx').values.tolist()
sentence = list(itertools.chain.from_iterable(df))
preprocessed_sentences = remove_null_sentence(text_preprocessing(sentences=sentence))
input_sentence = word_tokenizer(preprocessed_sentences)
result = word_count_per_document(input_sentence)
result.plot.line(figsize=(8,10),style='.-',)
