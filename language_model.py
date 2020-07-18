#Load library

import pandas as pd
from fastai.callbacks import hook_output
from fastai.text import *
import pickle
import torch
tok = SpacyTokenizer('en')
Training = False
#Load a ticket data
df = pd.read_excel('Tickets.xlsx')

#Preprocessing()
tokenizer = Tokenizer(pre_rules=[fix_html, replace_rep, replace_wrep, rm_useless_spaces], post_rules=[replace_all_caps, deal_caps], special_cases=[UNK,PAD,BOS])
#processor = [TokenizeProcessor(tokenizer=tokenizer,include_eos=True), NumericalizeProcessor(max_vocab=1000,min_freq=1)]
#data = (TextList.from_df(path='.',df=df, cols=0, processor=processor).split_by_rand_pct(0.2).label_for_lm().databunch(bs=32))
data=load_data(path='data/',file='databunch')

##Build or load a language model
learn = language_model_learner(data=data, arch=AWD_LSTM,config= {'emb_sz':100,'n_hid':500,'n_layers':3,'bidir':False,'tie_weights':True,'output_p':0.1,'out_bias':True},drop_mult=0.5,pretrained=False)
#learn.lr_find()
#learn.recorder.plot(return_fig=True)

if Training:
    learn.fit(epochs=200,lr=slice(1e-3,1e-4),wd=1e-6)
    model=learn.model.eval()
    learn.save_encoder('Encoder_model')
else:
    learn.load_encoder('Encoder_model')

model = learn.model.eval()
hidden=[model[0](i.cuda()) for i in data.dl()]
torch.save(torch.stack(hidden,dim=0),'ticket_representation.pth')
#learn.predict(text='we could not submit the order in  RCXO one of the TNS in view telephone',n_words=50,min_p=.8)