from transformers import BertModel, BertTokenizer
import re
import pickle
import torch
import esm
from tqdm import tqdm

Name = 'DNA-573_Train'
# Name = 'RNA-495_Train'
# Name = 'DRNA-1068_Train'

# local ESM2
model_path = './process_feature/esm2_t33_650M_UR50D/esm2_t33_650M_UR50D.pt'
model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
model.eval()
batch_converter = alphabet.get_batch_converter()

# data training data
data_dict = {}
data = open("../IPDLPre/Raw_data/protein/" + Name + ".txt", 'r').readlines()
for i in tqdm(range(len(data))):
    if data[i].startswith('>'):
        pid = data[i].strip()[1:]
        seq = [(pid, data[i+1].strip())]
        batch_labels, batch_strs, batch_tokens = batch_converter(seq)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        data_dict[pid] = (token_representations.squeeze(0)[1:-1, :], data[i+2].strip())

pickle.dump(data_dict, open("../IPDLPre/Dataset/DNA/esm_" + Name + ".pkl", 'wb'))
# pickle.dump(data_dict, open("../IPDLPre/Dataset/RNA/esm_" + Name + ".pkl", 'wb'))
# pickle.dump(data_dict, open("../IPDLPre/Dataset/DRNA/esm_" + Name + ".pkl", 'wb'))
