import numpy as np
import torch
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
from docutils.nodes import section
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
from RNN_decoder import RNN_Decoder

device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
nlp = spacy.load('en_core_web_sm')
pretrained=True
load_representation = False

def text_preprocessing(sentences:[str]):

    lema_text = new_lemmatization(sentences)
    input_text = list(tc.document(lema_text).remove_numbers().remove_stpwrds().remove_symbols().lower_all())
    return input_text

def lemmatization(sentences:[str], allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in sentences:
        doc = nlp(sent.lower())
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

def new_lemmatization(sentences:[str],allowed_postags=['NN','NNS', 'NNP','NNPS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ','JJ','JJR','JJS']):
    texts_out = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        tagged = nltk.pos_tag(tokens)
        pos_cleaned_sent = " ".join([token for (token, pos) in tagged if pos in allowed_postags])
        doc = nlp(pos_cleaned_sent)
        # Extract the lemma for each token and join
        texts_out.append(" ".join([token.lemma_ for token in doc]))
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
    # so this returns a list of listsd

def load_checkpoint(checkpoint_path, model, optimizer):
    """ loads state into model and optimizer and returns:
        epoch, model, optimizer
    """
    #model_path = 'models/seq2seq/without_batchnorm'
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(epoch, checkpoint['epoch']))
        return epoch,model, optimizer
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

def train(model, iterator, optimizer, criterion,epoch_id):

    model.train()
    epoch_loss = 0
    print_interval = 10
    for idx, batch_xy in enumerate(iterator):
        src = batch_xy[0]
        #trg_decoder = batch_xy[1]
        trg = batch_xy[1]
        pred, hidden = model(src)
        output = pred.permute(1,0,2).squeeze()
        loss = 0.
        #for i,p in enumerate(output):
        loss += criterion(output,trg.squeeze())
        loss /= output.size(0)
        # loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if idx % 5 == 0:
            print('Epoch : {}, sample {}/{} training loss = {}'.format(epoch_id,idx+1,len(iterator),epoch_loss / (idx+1)))
    return epoch_loss / len(iterator)

def generate_embeddings(model,sentences,Word2vec_model):
    model.eval()
    with torch.no_grad():
        preprocessed_sentences = remove_null_sentence(text_preprocessing(sentences=sentences))
        input_sentence = word_tokenizer(preprocessed_sentences)
    # model = model()
        output=[]
        for sentence in input_sentence:
            print(sentence)
            sent = []
            for word in sentence:
                sent.append(torch.from_numpy(Word2vec_model.wv.word_vec(word)).to(device))
            feature_tensor = torch.stack(sent).unsqueeze(0)
            output.append(model._modules.get('encoder')(feature_tensor).squeeze())
    return output

def testing(sentense,representation,model,Word2vec_model,df,processed_sentences,k=15):

    sentences=[]
    sentences.append(sentense)
    input_representation = torch.stack(generate_embeddings(model=model,sentences=sentences,Word2vec_model=Word2vec_model))
    cosine_distance = nn.CosineSimilarity(dim=0, eps=1e-6)
    distance=[]
    for i in representation:
        output = cosine_distance(input_representation[0], i)
        distance.append(output)
    distance = np.array(distance)
    index = distance.argsort()[-k:][::-1]
    for i in index:
        print(i,"------->",df.iloc[i,:].values[0],"------->",processed_sentences[i],"------->",distance[i].item())

def prepare_dataset(single_sentence,pretrained_model,dictionary,Batch_size=16):

    sentence=[]
    sentence.append(single_sentence)
    for sentence in sentence:
        sent=[]
        index_feature = torch.tensor(dictionary.doc2idx(sentence, unknown_word_index=random.randint(1, len(dictionary.token2id))),dtype=torch.long).to(device)
        #index.append(index_feature)#[:Max_len])
        for word in sentence:
            sent.append(torch.from_numpy(pretrained_model.wv.word_vec(word)).to(device))
        feature_tensor = torch.stack(sent)
        # if len(sentence)<Max_len:
        #     padded_tensor = torch.zeros(size=((Max_len-len(sentence)),Embedding_dim),requires_grad=True,device=device)
        #     feature.append(torch.cat((feature_tensor, padded_tensor)))
        # elif len(sentence)>=Max_len:
        #     feature.append(feature_tensor[:Max_len])
    index_tensor= index_feature#torch.stack(index)
    input_tensor=feature_tensor #torch.stack(feature)
    pairs =[(input_tensor, index_tensor)]
    train_iterator = torch.utils.data.DataLoader(dataset=pairs, batch_size=Batch_size)
    return train_iterator

df = pd.read_excel('Tickets.xlsx').values.tolist()
sentences = list(itertools.chain.from_iterable(df))
preprocessed_sentences = remove_null_sentence(text_preprocessing(sentences=sentences))
input_sentence = word_tokenizer(preprocessed_sentences)
# dictionary = Dictionary(input_sentence)
# dictionary.save("models/Word2vec/word2vec_dict_with_new_lemma.pkl")
dictionary = Dictionary.load("models/Word2vec/word2vec_dict_with_new_lemma.pkl")
#create_dictionary(sentences)
dictionary[0]
final_dict, number_dict = dictionary.token2id,dictionary.id2token #load_dictionary(path='models/Seq2Seq_LSTM/dict.pickle')
Word2vec_model = Word2Vec(input_sentence, size=100, window=5, min_count=1, workers=8,sg=1) #replace with bigram_sent,sentence for bigram model, unigram evaluation
 # print(Word2vec_model)
Word2vec_model.train(input_sentence,total_examples=len(preprocessed_sentences),epochs=10,compute_loss=True)
#Word2vec_model.init_sims(replace=True)
Word2vec_model.save("models/Word2vec/Word2Vec_with_new_lemma.model")
Word2vec_model = gensim.models.Word2Vec.load("models/Word2vec/Word2Vec_with_new_lemma.model")
## Hyper-parameter
INPUT_DIM = len(final_dict)
EMDEDDING_DIM = 100
HID_DIM = 200
N_LAYERS=1
BATCH_SIZE = 1
EPOCH = 20

def main():

    #input_batch, target_batch, pairs = make_batch(sentences=preprocessed_sentences,dictionary=dictionary)
    encoder = CNN_Encoder(input_size=EMDEDDING_DIM,hidden_size=HID_DIM,n_layers=N_LAYERS).to(device)
    n_enc_parms = sum([p.numel() for _, p in encoder.named_parameters() if p.requires_grad == True])
    print(encoder,n_enc_parms)
    decoder = RNN_Decoder(hidden_size=HID_DIM,output_size=INPUT_DIM,n_layers=N_LAYERS).to(device)
    n_dec_parms = sum([p.numel() for _, p in decoder.named_parameters() if p.requires_grad == True])
    print(decoder,n_dec_parms)
    model = CNN_LSTM(encoder, decoder).to(device)
    # encoder_optimizer = optim.Adam(encoder.parameters())
    # decoder_optimizer = optim.Adam(decoder.parameters())
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(PATH='models/CNN-LSTM', patience=10,verbose=True)
    if pretrained:
        epoch,model,optimizer =load_checkpoint('models/CNN-LSTM/train_loss-1.5566633939743042_epoch-0_checkpoint.pth.tar',model,optimizer)
    else:
        for i in range(EPOCH):
            for sentence in input_sentence :
                train_iterator = prepare_dataset(single_sentence=sentence, pretrained_model=Word2vec_model,dictionary=dictionary, Batch_size=BATCH_SIZE)
                train_loss = train(model=model,iterator=train_iterator,optimizer=optimizer,criterion=criterion,epoch_id=i)
            early_stopping(model,train_loss,optimizer,i)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        #save_checkpoint(Path='models/Seq2Seq_LSTM',encoder=encoder, encoder_optimizer=encoder_optimizer,epoch_id=i,train_loss=loss)

    if not load_representation:
        representations = generate_embeddings(model=model,sentences=sentences,Word2vec_model=Word2vec_model)
        torch.save(torch.stack(representations).to(device),'models/CNN-LSTM/representations.pth')
        print("total {} representation saved".format(len(representations)))
    else:
        representations = torch.load('models/CNN-LSTM/representations.pth')
    print("=============== Representation Loaded =============")

    print("================Testing========================")
    df=pd.read_excel('Tickets.xlsx')
    pd.set_option('display.max_columns', 500)
    while True:
        input_ticket = input('Type Ticket text : ')
        #result=
        testing(sentense=input_ticket,representation=representations,model=model,Word2vec_model=Word2vec_model,df=df,processed_sentences=preprocessed_sentences,k=30)
        #print(result.values)
if __name__ == "__main__":
    main()
