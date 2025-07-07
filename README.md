IPDLPre: Protein-DNA binding site prediction

IPDLPre is primarily dependent on a large-scale pre-trained protein language model ESM-2,protT5 and protbert implemented using HuggingFace's Transformers and PyTorch.

Generate Features

After completing the steps above, run the following command to generate features and save them in the ./feature directory:

python ./process_feature/process_feature.py
Alternatively, you can download the features directly from our cloud drive: Google Drive Link.
