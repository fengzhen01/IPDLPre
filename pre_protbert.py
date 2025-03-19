from transformers import BertModel, BertTokenizer
import re
import pickle
import torch
from tqdm import tqdm

# ProtBert
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, cache_dir='./cache_model/')
pretrain_model = BertModel.from_pretrained("Rostlab/prot_bert", cache_dir='./cache_model/')

def get_protein_features(seq):
   sequence_Example = ' '.join(seq)
   sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
   encoded_input = tokenizer(sequence_Example, return_tensors='pt')
   last_hidden = pretrain_model(**encoded_input).last_hidden_state.squeeze(0)[1:-1,:]
   return last_hidden.detach()

def data_pkl_generator(root, saver):
   data = open(root, 'r').readlines()
   data_dict = {}
   for i in tqdm(range(len(data))):
       if data[i].startswith('>'):
           seq = data[i+1].strip()
           label = data[i+2].strip()
           data_dict[data[i].strip()[1:]] = (get_protein_features(seq), label)
   pickle.dump(data_dict, open(saver, 'wb'))


Name = 'DRNA-1068_Train'
file_path = "../IPDLPre/Raw_data/DNA/" + Name + ".txt"
saver_path = "../IPDLPre/Dataset/DRNA-1068/protbert_" + Name + ".pkl"

data_pkl_generator(file_path, saver_path)
