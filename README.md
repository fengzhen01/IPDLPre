# Protein-DNA Binding Sites Prediction via Integrating Pretrained Large Language Models and Contrastive Learning
This repository provides the implementation of the IPDLPre framework for predicting protein–DNA binding sites, with additional support for protein–RNA binding site prediction.  
IPDLPre is built upon three large-scale, pre-trained protein language models: ESM-2, ProtT5, and ProtBert. These models are integrated using Hugging Face's Transformers and PyTorch.  
Please ensure that all required dependencies are installed before running the code.   
ESM-2: https://huggingface.co/facebook/esm2_t12_650M_UR50D  
ProtT5: https://huggingface.co/Rostlab/prot_t5_xl_uniref50  
ProtBert: https://huggingface.co/Rostlab/prot_bert  

# 1. Requirements

# 2. Datasets

# 3. How to use

## 3.1 Set up environment for ProtTrans 

## 3.2 Train and test

### 3.2.1 Extract features

### 3.2.2 Train and test

## 3.3 Only For nucleotide binding residues prediction purpose

### Generate Features

After completing the steps above, run the following command to generate features and save them in the ./feature directory:

python ./process_feature/process_feature.py
Alternatively, you can download the features directly from our cloud drive: Google Drive Link.
