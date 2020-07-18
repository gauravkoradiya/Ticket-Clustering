import collections
import csv
import itertools
import spacy
import pandas as pd
import textcleaner as tc
from docutils.nodes import section
import gensim.models
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize,word_tokenize
import gensim
import string
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from random import choice
nlp = spacy.load('en_core_web_sm')
#doc = nlp('The John present invention relates to semiconductor design technologies; and, more particularly, to a semiconductor memory device which is provided with an internal power generator capable of increasing a cell retention time during a self refresh mode and a refresh unit capable of minimizing power consumption by adjusting a refresh period depending on a level of an internal power.A semiconductor memory device includes a charge pumping circuit, a level sensor, an oscillator, and a pumping control signal generator. The charge pumping circuit performs a negative-pumping operation to an external power in order to generate an internal voltage having a level lower than the external power. The level sensor senses a level of the internal voltage corresponding to a level of an adjusted reference voltage during a refresh mode. The oscillator generates a period signal in response to a sensing signal of the level sensor. The pumping control signal generator controls the operation of the charge pumping circuit in response to the period signal.')
#
# ####Basic document field exploration####

#print(dir(doc))
#print(doc.doc)
#print(list(doc.sents))
#print(list(doc.noun_chunks))
#---Subject, Object, Date object detection
#for np in doc.noun_chunks:
#    print(np.text, np.root.dep_, np.root.head.text)
#print(doc.sentiment)
#print(doc.vocab)
#for token in doc:
#    print('"' + token.text + '"')

#--Find POS info
#for word in list(doc.sents)[0]:  
#    print(word, word.pos_)

#--Find phrases
#print(Counter(list(doc.noun_chunks)).most_common(6))

    
####Cleanup####
    
#define some parameters  
# noisy_pos_tags = ["PROP"]
# min_token_length = 2

#Function to check if the token is a noise or not  
# def isNoise(token):
#     is_noise = False
#     if token.pos_ in noisy_pos_tags:
#         is_noise = True
#     elif token.is_stop == True:
#         is_noise = True
#     elif len(token.string) <= min_token_length:
#         is_noise = True
#     return is_noise
#
# def cleanup(token, lower = True):
#     if lower:
#        token = token.lower()
#     return token.strip()
#
# #---top unigrams used in the reviews
# #cleaned_list = [cleanup(word.string) for word in doc if not isNoise(word)]
# #print(collections.Counter(cleaned_list).most_common(5))
#
# #---Entity detection
# #labels = set([w.label_ for w in doc.ents])
# #for label in labels:
# #    entities = [cleanup(e.string, lower=False) for e in doc.ents if label==e.label_]
# #    entities = list(set(entities))
# #    print(label,entities)
#
#
# # extract all review sentences that contains the term - hotel
# #hotel = [sent for sent in document.sents if 'hotel' in sent.string.lower()]
# ##---create dependency tree
# #sentence = hotel[2] for word in sentence:
# #print(word, ': ', str(list(word.children)))
#
#
# # check all adjectives used with a word
# #def pos_words (sentence, token, ptag):
# #    sentences = [sent for sent in sentence.sents if token in sent.string]
# #    pwrds = []
# #    for sent in sentences:
# #        for word in sent:
# #            if character in word.string:
# #                   pwrds.extend([child.string.strip() for child in word.children
# #                                                      if child.pos_ == ptag] )
# #    return Counter(pwrds).most_common(10)
#
# #pos_words(document, 'hotel', “ADJ”)
#
#
# #from spacy import displacy
# #
# #doc = nlp('I just bought 2 shares at 9 a.m. because the stock went up 30% in just 2 days according to the WSJ')
# #displacy.render(doc, style='ent', jupyter=True)
#
#
# ########################################## S C R A P P I N G ##########################################
#
# # try:
# #     from googlesearch import search
# # except ImportError:
# #     print("No module named 'google' found")
# # from bs4 import BeautifulSoup
# # from bs4.element import Comment
# #
# # def get_text_from_html(body):
# #     soup = BeautifulSoup(body, 'html.parser')
# #     texts = soup.findAll(text=True)
# #     visible_text = filter(visible_tags, texts)
# #     return u" ".join(t.strip() for t in visible_text)
# #
# # def visible_tags(element):
# #     if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
# #         return False
# #     if isinstance(element, Comment):
# #         return False
# #     return True
#
#
# #
# #try:
# #    header_param = {"Content-Type":" application/x-www-form-urlencoded;charset=utf-8",
# #              'User-Agent' : 'Mozilla/5.0 (X11; Linux x86_64; rv:10.0) Gecko/20100101 Firefox/10.0',
# #              'Accept' : 'text/html, application/xhtml+xml, application/xml;q=0.9, */*;q=0.8',
# #              'Accept-Charset' : 'utf-8, iso-8859-1;q=0.5',
# #              'Accept-Encoding' : 'none',
# #              'Accept-Language' : 'en-US,en;q=0.5',
# #              'Connection' : 'keep-alive'}
# #
# #    query = "Virtual Machine technology"
# #    urls =[]
# #    for j in search(query, tld="com", num=3, stop=1, pause=2):
# #        urls.append(j)
# #    url_data = urllib.request.urlopen(urllib.request.Request(urls[0], headers = header_param)).read()
# #    text_body = get_text_from_html(url_data)
# #    print(text_body)
# #
# #except urllib.error.HTTPError as err:
# #    if (err!=200):
# #        print("Cannot Reach Website.")
#
#
# ########################################## W O R D  C L O U D ##########################################
# try:
#     from googlesearch import search
# except ImportError:
#     print("No module named 'google' found")
# #from newsplease import NewsPlease
# from nltk.tokenize import sent_tokenize,word_tokenize
# import gensim
# import string
# from nltk.corpus import stopwords
# from random import choice
#
#
# # In[4]:
#
#
# # term to be searched
# query = 'Virtual Machine Technology'
# #query = query + " Technology"
#
# user_agent_list = [
#    #Chrome
#     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
#     'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
#     'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
#     'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
#     'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
#     'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
#     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
#     'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
#     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
#     'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
#     #Firefox
#     'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
#     'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
#     'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
#     'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
#     'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
#     'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
#     'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0)',
#     'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
#     'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
#     'Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko',
#     'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
#     'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
#     'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)'
# ]
#
# def random_headers():
#     return {'User-Agent': choice(user_agent_list)}
#
# header_param=random_headers()
# #header_param = {"Content-Type":" application/x-www-form-urlencoded;charset=utf-8",
# #          'User-Agent' : 'Mozilla/5.0 (X11; Linux x86_64; rv:10.0) Gecko/20100101 Firefox/10.0',
# #          'Accept' : 'text/html, application/xhtml+xml, application/xml;q=0.9, */*;q=0.8',
# #          'Accept-Charset' : 'utf-8, iso-8859-1;q=0.5',
# #          'Accept-Encoding' : 'none',
# #          'Accept-Language' : 'en-US,en;q=0.5',
# #          'Connection' : 'keep-alive'}
# urls =[]
# for j in search(query, tld='com', lang='en', num=3, start=0, stop=None, pause=2.0):
#     urls.append(j)
# #    url_data = urllib.request.urlopen(urllib.request.Request(j, headers = header_param)).read()
# #    input_text = get_text_from_html(url_data)
# #    print('###########\n',j)
# #    print(input_text, '\n')
#
# print(urls)
#
#

# In[55]:


#sentence sets up unigrams, bigram_sent sets up bigrams, trigram_sent sets up trigramms

#for i in range (0,len(urls)):
##parse the articles and extract
#    article = NewsPlease.from_url(urls[i])
#    input_text=str(article.text)
df = pd.read_excel('Tickets.xlsx').values.tolist()
sentence = list(itertools.chain.from_iterable(df))

def sentence_tokenizer(sentences:str):
    raw_sentences = []
    doc = nlp(sentences)
    for i, token in enumerate(doc.sents):
        print('-->Sentence %d: %s' % (i, token.text))
        raw_sentences.append(token.text)
    return raw_sentences

def text_preprocessing(sentences:[str]):

    input_text = list(tc.document(sentences).remove_numbers().remove_stpwrds().remove_symbols().lower_all())
    lema = lemmatization(sentences=input_text)
    return lema

def lemmatization(sentences:[str], allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in sentences:
        doc = nlp(sent.lower())
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
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

preprocessed_sentences = remove_null_sentence(text_preprocessing(sentences=sentence))
input_sentence = word_tokenizer(preprocessed_sentences)
dictionary = Dictionary(input_sentence)
#set up bigram model, store results in a list of lists

# bigram_model = gensim.models.Phrases(input_sentence,min_count=1,threshold=10)
# bigram_sent= [bigram_model[sent] for sent in input_sentence]
# print(bigram_sent)
#
# #set up trigram model, store results in a list of lists
# trigram_model = gensim.models.Phrases(bigram_model[bigram_sent],threshold=10)
# trigram_sent= [trigram_model[bigram_model[sent]] for sent in input_sentence]
# print(trigram_sent)

# model = Word2Vec(input_sentence, size=100, window=5, min_count=1, workers=8,sg=1) #replace with bigram_sent,sentence for bigram model, unigram evaluation
# # #print(model)
# model.train(input_sentence,total_examples=len(input_sentence),epochs=10)
# model.init_sims(replace=True)
# model.save("models/Word2vec/Word2Vec.model")
model=Word2Vec.load("models/Word2vec/Word2Vec.model")
# model.wv.save_word2vec_format('models/Word2vec/model.txt', binary=False)
frequency_dict={}

def save_dict_to_file(model):
    with open('models/Word2vec/freqency_dict.txt', 'w') as f:
        f.writelines('{} {}\n'.format(word, freq.count) for word, freq in model.wv.vocab.items())
    print('== Dictionary Saved ===')
save_dict_to_file(model=model)


def load_dict_from_file(path):
    frequency_dict = {}
    with open(path, 'r') as raw_data:
        for item in raw_data:
            key,value = item.rstrip().split(' ',1)
            frequency_dict[key] = int(value)
    return frequency_dict

load_dict_from_file(path = 'models/Word2vec/freqency_dict.txt')

result = model.wv.most_similar(positive='bvoip',topn=20)
print(result)
print(type(model.wv.word_vec('bvoip')))
print()

# In[89]

# stopset= list(stopwords.words('english'))
# stopset.extend(['from', 'subject', 're', 'edu', 'use', 'many', 'can_be', 'like','see', 'to_be','did_not','have'])
# #if we can find a generic set of words that we do not need we can incorporate within this set, qwe can easily extended.
# clean_result = []
# for m in result:
#     word = [i for i in m if str(i).lower() not in stopset]
#     #eliminate the cosine similarity score as well
#     if len(word)>1:
#         clean_result.append(word)
# print(clean_result)
# print(len(clean_result))


# In[124]:


#combining results as a string for ner detection

#string=''
#for i in range (0, len(clean_result)):
#    string= str(string) + ' ' + str(clean_result[i][0])
#   
#print(string)
#
#
## In[128]:
#
#
##Named entity recognition using generic NLTK (not very accurate)
#for sent in nltk.sent_tokenize(string):
#   for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
#      if hasattr(chunk, 'label'):
#         print(chunk.label(), ' '.join(c[0] for c in chunk))
#
#
## In[131]:
#
#
##ner using spacy (much more accurate, closest we can get without custom-training a model for NER)
#nlp=spacy.load('en')
#doc=nlp(string)
#
#for ent in doc.ents:
#    print(ent.text, ent.label_)
# import gensim
#
# # Load pre-trained Word2Vec model.
# model = gensim.models.Word2Vec.load("models/Word2vec/Word2Vec.model")
#
# dictionary = Dictionary(input_sentence)
# dictionary.save("tag_dictionary.pkl")
# dictionary = Dictionary.load('models/Word2vec/word2vec_dict.pkl')
