import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pygments.lexer import words
from setuptools.command.dist_info import dist_info
from spacy.syntax.nn_parser import precompute_hiddens
from torch.autograd import Variable
import pandas as pd
import random
import itertools
import nltk
from nltk.corpus import stopwords
import random
from collections import Counter
import string
import re
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
from nltk.corpus import subjectivity
#pd.options.display.max_colwidth = 100
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
nlp = spacy.load('en_core_web_sm')
# # Load pre-trained Word2Vec model.
# Word2vec_model = gensim.models.Word2Vec.load("models/Word2vec/Word2Vec.model")
# word2vec_dict = Dictionary.load('models/Word2vec/Word2Vec_dict.pkl')
# word2vec_dict.add_documents([['EOS','SOS']])
stop_words = set(stopwords.words('english'))
custom_stop_words = {'please'}
final_stop_words = stop_words.union(custom_stop_words)
add_tok = {'UNK':1, 'PAD':0, 'SOS':2,'EOS':3}
pretrained=False
load_representation = False

def create_dictionary(sentences):

    translator = str.maketrans('', '', string.punctuation)
    full_sentense=" ".join(sentences).translate(translator).lower().split()
    word_list = [word for word in set(full_sentense) if word not in final_stop_words]
    word_dict = {w : i for i, w in enumerate(word_list)}
    final_dict = {**word_dict,**add_tok}
    #Save Dict
    pickle_out = open("models/Seq2Seq_LSTM/dict.pickle","wb")
    pickle.dump(final_dict, pickle_out)
    pickle_out.close()

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
    # so this returns a list of lists


def load_dictionary(path):
    final_dict= pickle.load(open(path, "rb"))
    number_dict = {i : w for w, i in final_dict.items()}
    return final_dict,number_dict

def getCleanText(clean_text):
    clean_text_result = re.sub(r"[{}–!='™‘’â€˜|?,-:@#%&$/1234567890()]+/*", " ", clean_text)
    return clean_text_result


def make_batch(sentences,dictionary):
    input_batch = []
    target_batch = []
    pairs=[]
    for index,sen in enumerate(sentences):
        s=[]
        #clean_sen=lemmatization(sen).strip()
        #sen= 'SOS '+clean_sen+' EOS'
        word = sen.split()
        #process input sentence
        # for n in word:
        #     if n not in list(dictionary.keys()):
        #         continue
        #     else:
        #         s.append(dictionary[n])
        s=dictionary.doc2idx(word,unknown_word_index=random.randint(1,len(dictionary.token2id)))
        src = torch.tensor(s[1:-1],dtype=torch.long).to(device)
        trg= torch.tensor(s[2:],dtype=torch.long).to(device)

        # if src.shape[0] == 0:
        #     print(index)
        input_batch.append(src)
        target_batch.append(trg)
        pairs.append((src,trg))
    return input_batch, target_batch,pairs

# torch.save(input_batch,'input.pth')
# torch.save(target_batch,'target.pth')
# input_tensor = torch.load('input.pth')
# target_tensor = torch.loa bd('target.pth')


teacher_forcing_ratio = 0.5
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=30):

    encoder.train()
    decoder.train()

    #encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    seq_length = input_length if input_length<max_length else max_length
    for ei in range(seq_length):
        input_embedding = Word2vec_model.wv.word_vec(number_dict[input_tensor[ei].item()])
        encoder_output, encoder_hidden = encoder(torch.from_numpy(input_embedding).float().to(device)) #encoder_hidden)
        #encoder_outputs[ei] = encoder_output[0, 0]

    # if target_length < 2:
    #     decoder_input = torch.tensor([[add_tok['SOS']]], device=device)
    # else:

    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_input = Word2vec_model.wv.word_vec(number_dict[input_tensor[di].item()])
            decoder_output, decoder_hidden = decoder(torch.from_numpy(decoder_input).float().to(device), decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
              # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        decoder_input = Word2vec_model.wv.word_vec(number_dict[input_tensor[0].item()])
        for di in range(target_length):
            decoder_output, decoder_hidden= decoder(torch.from_numpy(decoder_input).float().to(device), decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input_token = topi.squeeze().detach()  # detach from history as input
            decoder_input = Word2vec_model.wv.word_vec(number_dict[decoder_input_token.item()])
            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

def trainIters(encoder, decoder,pairs,encoder_optimizer,decoder_optimizer,criterion, print_every=1000,epoch_id=1):

    print_loss_total = 0  # Reset every print_every
    training_pairs = pairs

    for iter in range(len(training_pairs)):
        pair = training_pairs[iter]
        input_tensor = pair[0]
        target_tensor = pair[1]
        if len(input_tensor)>0:
            loss = train(input_tensor, target_tensor, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('Epoch : %d || Iteration : %d || completeness : %d%% || Training_Loss : %.4f' % (epoch_id,iter, iter / len(training_pairs)* 100, print_loss_avg))
    print("Loss during {} epoch : {}".format(epoch_id,print_loss_total/len(training_pairs)))
    return print_loss_total/len(training_pairs)

def evaluate(encoder, sentence):

    encoder.eval()
    with torch.no_grad():
        input_tensor=sentence
        input_length = input_tensor.shape[0]
        #encoder_hidden = encoder.initHidden()

        #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        encoder_outputs=[]
        for ei in range(input_length):
            input_embedding = Word2vec_model.wv.word_vec(number_dict[input_tensor[ei].item()])
            encoder_output, encoder_hidden = encoder(torch.from_numpy(input_embedding).float().to(device))#,encoder_hidden)
            #encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs.append(encoder_output[0, 0])

        last_hidden = encoder_hidden

        #decoder_input = torch.tensor([[add_tok['SOS']]], device=device)  # SOS
        # decoded_words = []
        #
        # for di in range(max_length):
        #     decoder_output, decoder_hidden= decoder(decoder_input, decoder_hidden)
        #     topv, topi = decoder_output.data.topk(1)
        #     if topi.item() == add_tok['EOS']:
        #         decoded_words.append('EOS')
        #         break
        #     else:
        #         decoded_words.append(number_dict[topi.item()])
        #     decoder_input = topi.squeeze().detach()

        return last_hidden,torch.stack(encoder_outputs,dim=0)

def evaluateRandomly(encoder,sentences,dictionary,method=None):
    input_batch,_,_= make_batch(sentences,dictionary)
    temp=[]
    if method=='middle_last':
        for i in input_batch:
            if i.size(0) == 0:
                print("=============*****************================********************=================")
                representation = torch.zeros(400).to(device)
                print("=============*****************================********************=================")
            elif i.size(0) < 5 and i.size(0)>0:
                print('>', i)
                last_hidden, embeddings = evaluate(encoder, i)
                zero_vec = torch.zeros(200).to(device)
                print(embeddings.shape)
                representation = torch.cat((zero_vec,last_hidden.view(-1))).flatten().to(device)
                print('======================================================================================================')
            else:
                print('>',i)
                last_hidden,embeddings= evaluate(encoder, i)
                print(embeddings.shape)
                print('')
                indices = torch.tensor([int(embeddings.size(0) / 2)-1, int(embeddings.size(0)) - 1]).to(device)
                representation=torch.index_select(embeddings, 0, indices).flatten().to(device)
                print('======================================================================================================')
            temp.append(representation)
    else:
        for i in input_batch:
            if i.size(0) == 0:
                representation = torch.zeros(200).to(device)
            else:
                last_hidden, embeddings = evaluate(encoder, i)
                representation = last_hidden.view(-1)
            temp.append(representation)

    return temp

def save_checkpoint(Path,encoder,encoder_optimizer,epoch_id,train_loss):
    '''Saves model when validation loss decrease.'''
    SAVE_DIR = Path
    # #MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'model.pt')

    torch.save({
        'epoch': epoch_id,
        'state_dict': encoder.state_dict(),
        'optimizer': encoder_optimizer.state_dict(),
        'train_loss': train_loss,
    }, os.path.join(SAVE_DIR ,'train_loss-{:.4f}_epoch-{}_checkpoint.pth.tar'.format(train_loss,epoch_id)))
    print("model saved at : {}".format(SAVE_DIR,':train_loss-{:.4f}_epoch-{}_checkpoint.pth.tar'.format(train_loss,epoch_id)))

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


def testing(sentense,representation,dictionary,encoder,df,processed_sentences,k=15):
    sentences=[]
    sentences.append(sentense)
    preprocessed_sentences = remove_null_sentence(text_preprocessing(sentences=sentences))
    print(preprocessed_sentences)
    input_representation = evaluateRandomly(encoder=encoder,sentences=preprocessed_sentences,dictionary=dictionary)
    cosine_distance = nn.CosineSimilarity(dim=0, eps=1e-6)

    distance=[]
    for i in representation:
        output = cosine_distance(input_representation[0], i)
        distance.append(output)
    distance = np.array(distance)
    index = distance.argsort()[-k:][::-1]
    for i in index:
        print(i,"------->",df.iloc[i,:].values[0],"------->",processed_sentences[i],"------->",distance[i].item())
    #return df.iloc[index, :]
#
# def mixup_bert_word2vec_embedding(bert_file_path,word2vec_file_path):
#
#     with open(file=bert_file_path, mode='rb') as handle:
#         bert_embeddings = pickle.load(handle)
#     with open(file=word2vec_file_path, mode='rb') as handle:
#         word2vec_embeddings = pickle.load(handle)

#     print(type(bert_embeddings),type(word2vec_embeddings))
#     return torch.from_numpy(np.concatenate((bert_embeddings,word2vec_embedding),axis=1)).to(device)

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
result = Word2vec_model.wv.most_similar(positive='bvoip',topn=20)
Word2vec_model.init_sims(replace=True)
# Word2vec_model.save("models/Word2vec/Word2Vec_with_new_lemma.model")
Word2vec_model = gensim.models.Word2Vec.load("models/Word2vec/Word2Vec_with_new_lemma.model")
## Hyper-parameter
INPUT_DIM = len(final_dict)
HID_DIM = 200
EPOCH = 100

def main():

    input_batch, target_batch, pairs = make_batch(sentences=preprocessed_sentences,dictionary=dictionary)
    encoder = en.EncoderRNN(hidden_size=HID_DIM).to(device)
    n_enc_parms = sum([p.numel() for _, p in encoder.named_parameters() if p.requires_grad == True])
    print(encoder,n_enc_parms)
    decoder = dc.DecoderRNN(hidden_size=HID_DIM, output_size=INPUT_DIM).to(device)
    n_dec_parms = sum([p.numel() for _, p in decoder.named_parameters() if p.requires_grad == True])
    print(decoder,n_dec_parms)
    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(PATH='models/Seq2Seq_LSTM', patience=10,verbose=True)

    if pretrained:
        epoch,encoder,encoder_optimizer =load_checkpoint('models/Seq2Seq_LSTM/train_loss-0.04279935139991392_epoch-4_checkpoint.pth.tar',encoder,encoder_optimizer)
    else:
        for i in range(EPOCH):
            train_loss = trainIters(encoder,decoder,pairs, print_every=5,epoch_id=(i+1),encoder_optimizer=encoder_optimizer,decoder_optimizer=decoder_optimizer,criterion=criterion)
            early_stopping(encoder,train_loss,encoder_optimizer,i)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        #save_checkpoint(Path='models/Seq2Seq_LSTM',encoder=encoder, encoder_optimizer=encoder_optimizer,epoch_id=i,train_loss=loss)

    if not load_representation:
        representations = evaluateRandomly(encoder=encoder,sentences=preprocessed_sentences,dictionary=dictionary)
        # with open("models/Seq2Seq_LSTM/representations.pkl", "wb") as handle:
        #     pickle.dump(representation, handle)
        torch.save(torch.stack(representations).to(device),'models/Seq2Seq_LSTM/representations.pth')
        print("total {} representation saved".format(len(representations)))
    else:
        # with open("models/Seq2Seq_LSTM/representations.pkl", "rb") as handle:
        #     representation = pickle.load(handle)
        representations = torch.load('models/Seq2Seq_LSTM/representations.pth')
    print("=============== Representation Loaded =============")
    #mixed_representation = mixup_bert_word2vec_embedding(bert_file_path='models/BERT/Bert_representation.pickle',word2vec_file_path='models/Seq2Seq_LSTM/representations.pkl')
    #print(mixed_representation.shape)
    print("================Testing========================")
    df=pd.read_excel('Tickets.xlsx')
    pd.set_option('display.max_columns', 500)
    while True:
        input_ticket = input('Type Ticket text : ')
        #result=
        testing(sentense=input_ticket,representation=representations,dictionary=dictionary,encoder=encoder,df=df,processed_sentences=preprocessed_sentences,k=20)
        #print(result.values)
if __name__ == "__main__":
    main()

