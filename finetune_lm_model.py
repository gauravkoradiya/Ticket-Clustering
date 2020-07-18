from docutils.nodes import legend
from fastai.text import *
from sklearn.model_selection import train_test_split
import pandas as pd
from fastai.callbacks import *

#Load a ticket data
df = pd.read_excel('Tickets.xlsx')

def training_phase(df:DataFrame,split_size:int,text_cols:Collection['str'], data_saved:bool, model_saved:bool, data_save_path:Path,learner_save_path:Path):
    if not data_saved:
        #Train and validation split
        print("*")
        train_df, val_df = train_test_split(df, test_size=split_size,random_state=42)
        print("**")
        # Language model data
        data_lm = TextLMDataBunch.from_df(path='',train_df=train_df,valid_df=val_df,include_eos=True,include_bos=True,text_cols=text_cols)
        print("***")
        data_lm.save(data_save_path/'data_lm_export.pkl')
    else:
        data_lm=load_data(path=data_save_path,file='data_lm_export.pkl',bs=16)


    if not model_saved:
        learn = language_model_learner(data=data_lm,arch=Transformer,drop_mult=0.5,pretrained=False)#pretrained_fnames=['transformer.pth','itos_tfmer.pkl'])
        learn.fit(epochs=2,lr=slice(1e-3,1e-4,1e-2))
        learn.export(Path('../models/learner.pkl'))
    else:
        learn = load_learner(learner_save_path, 'learner.pkl')

    return learn,data_lm

learn,lm = training_phase(df,split_size=0.2,text_cols=['Problem Description'],data_saved=True,model_saved=True,data_save_path='data',learner_save_path='models')
print("success")
def testing_phase(df:DataFrame,leaner_path:Union[Path,str],learner_file:str,vocab_path:Union[Path,str]):

    tokenizer = Tokenizer(pre_rules=[fix_html, replace_rep, replace_wrep, rm_useless_spaces],
                      post_rules=[replace_all_caps, deal_caps],
                      special_cases=[UNK, PAD, BOS,EOS])
    vocab=Vocab.load('data/vocab')
    learn = load_learner(leaner_path,learner_file)
    model = learn.model[0].eval()
    tokens=tokenizer._process_all_1(df.values.tolist())
    input = torch.Tensor([vocab.stoi[i] for i in tokens]).long().unsqueeze(0).cuda()
    output=model(input)[0][0]
    # output = torch.stack(hidden,dim=0)
    return output

output=testing_phase(df=df,leaner_path='models',learner_file='learner.pkl',vocab_path='data/vocab')










