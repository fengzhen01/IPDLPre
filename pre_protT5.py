from transformers import T5Tokenizer, T5EncoderModel
import re
import pickle
import torch
from tqdm import tqdm

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
pretrain_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")


def get_protein_features(seq):
   sequence_Example = ' '.join(seq)
   sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
   encoded_input = tokenizer(sequence_Example, return_tensors='pt')
   with torch.no_grad():
       last_hidden = pretrain_model(**encoded_input).last_hidden_state.squeeze(0)
   return last_hidden[1:].detach()

def data_pkl_generator(root, saver):
   data = open(root, 'r').readlines()
   data_dict = {}
   for i in tqdm(range(len(data))):
       if data[i].startswith('>'):
           seq = data[i+1].strip()
           label = data[i+2].strip()
           data_dict[data[i].strip()[1:]] = (get_protein_features(seq), label)
   pickle.dump(data_dict, open(saver, 'wb'))


# Name = 'DNA-129_Test'
Name = 'DNA-129_Test'
file_path = "../IPDLPre/Raw_data/DNA/" + Name + ".txt"
saver_path = "../IPDLPre/Dataset/DRNA-1068/protT5_" + Name + ".pkl"

data_pkl_generator(file_path, saver_path)
