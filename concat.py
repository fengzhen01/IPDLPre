import pickle
import torch
from torch.nn.utils.rnn import pad_sequence


# Define functions for concatenating features
def merge_features(esm_data_path, protbert_data_path, protT5_data_path, output_path):
    # Load ESM2, ProtBert, ProtT5 data
    esm_data = pickle.load(open(esm_data_path, 'rb'))
    protbert_data = pickle.load(open(protbert_data_path, 'rb'))
    protT5_data = pickle.load(open(protT5_data_path, 'rb'))

    combined_data = {}

    for key in esm_data.keys():
        # 取出各个模型的特征
        esm_feature = esm_data[key][0]
        protbert_feature = protbert_data[key][0]
        protT5_feature = protT5_data[key][0]

        if esm_feature.shape[0] == protbert_feature.shape[0] == protT5_feature.shape[0]:
            combined_feature = torch.cat((esm_feature, protbert_feature, protT5_feature), dim=-1)

            label = esm_data[key][1]

            combined_data[key] = (combined_feature, label)
        else:
            print(f"Skipping {key} due to mismatched lengths.")

    with open(output_path, 'wb') as f:
        pickle.dump(combined_data, f)

# 路径配置
esm_data_root_Val = '../IPDLPre/Dataset/RNA/esm_RNA-495_Train.pkl'
protbert_data_root_Val = '../IPDLPre/Dataset/RNA/protbert_RNA-495_Train.pkl'
protT5_data_root_Val = '../IPDLPre/Dataset/RNA/protT5_RNA-495_Train.pkl'


three_data_root_Val = '../IPDLPre/Dataset/concat/DRNA/three_data_Train.pkl'
# two_data_root_Val = '../IPDLPre/Dataset/RNA3/two_data_Val.pkl'

merge_features(esm_data_root_Val, protbert_data_root_Val, protT5_data_root_Val, three_data_root_Val)

print("Feature merging complete!")