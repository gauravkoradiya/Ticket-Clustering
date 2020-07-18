import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM,BertConfig
import logging
import torch
import spacy
import itertools
import pickle
import pandas as pd

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
nlp = spacy.load('en_core_web_sm')
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('models/BERT/bert-base-uncased-vocab.txt')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('models/BERT/bert-base-uncased.tar.gz').cuda()

def lemmatization(sent, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    # texts_out = []
    # for sent in texts:
    doc = nlp(sent.lower())
    return " ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags])
    # return texts_out

def add_special_tok(sentence):

    marked_text ="[CLS] "+lemmatization(sentence).strip()+" [SEP]"
    return marked_text

def text_tokenization(sentences):

    return [tokenizer.tokenize(add_special_tok(sentence)) for sentence in sentences]

def numericalization(tokenized_text):

    return [tokenizer.convert_tokens_to_ids(text) for text in tokenized_text]

def gen_segmentationID(tokenized_text):

    return [[i] * len(tokenized_text[i]) for i in range(len(tokenized_text))]


def generate_representation(sentences:[str], model):

    tokenized_text = text_tokenization(sentences)
    indexed_tokens = numericalization(tokenized_text)

    model.eval()
    sentences_representation = []
    word_representation = []
    for i, j in enumerate(indexed_tokens):

        token = torch.Tensor([j]).long().to(device)
        segment = torch.Tensor([[0] * token.shape[1]]).long().to(device)
        print("input :{},{}".format(token, segment))
        with torch.no_grad():
            encoded_layers, _ = model(token, segment)
            print("Number of hidden units:", len(encoded_layers[0][0][0]))
            # Convert the hidden state embeddings into single token vectors
            # Holds the list of 12 layer embeddings for each token
            # Will have the shape: [# tokens, # layers, # features]
            token_embeddings = []
            batch_i = 0  # Always one batch will be there for first sentence
            # For each token in the sentence...
            for token_i in range(token.shape[1]):
                # Holds 12 layers of hidden states for each token
                hidden_layers = []
                # For each of the 12 layers...
                for layer_i in range(len(encoded_layers)):
                    # Lookup the vector for `token_i` in `layer_i`
                    vec = encoded_layers[layer_i][batch_i][token_i]

                    hidden_layers.append(vec)

                token_embeddings.append(hidden_layers)
            print("full vector embeddings generated")
            # Sanity check the dimensions:
            print("Number of tokens in sequence:", len(token_embeddings))
            print("Number of layers per token:", len(token_embeddings[0]))
            # Stores the token vectors, with shape [22 x 768]
            token_vecs_sum = []
            # For each token in the sentence...
            for token in token_embeddings:
                # Sum the vectors from the last four layers.
                # sum_vec = torch.sum(torch.stack(token)[-4:], 0)
                sum_vec = token[11]
                # Use `sum_vec` to represent `token`.
                token_vecs_sum.append(sum_vec)
            word_representation.append(torch.stack(token_vecs_sum, dim=0))
            sentences_representation.append(torch.mean(torch.stack(token_vecs_sum, dim=0), 0))
            print("=============================================================================================================================")

    torch.save(torch.stack(sentences_representation, dim=0), 'models/BERT/bert_representation.pth')
    with open('models/BERT/word_representations.pkl', 'wb') as f:
        pickle.dump(word_representation, f)
    print("Sentence embeddings stored at : models/BERT/bert_representation.pth ")
    return word_representation,torch.stack(sentences_representation, dim=0)

df = pd.read_excel('Tickets.xlsx').values.tolist()
sentences = list(itertools.chain.from_iterable(df))
word_representation,sent_representation=generate_representation(sentences,model)

sent_representation=torch.load('models/BERT/bert_representation.pth')
with open('models/BERT/word_representations.pkl', 'rb') as f:
        word_representation=pickle.load(f)