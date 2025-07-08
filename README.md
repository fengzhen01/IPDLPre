# Protein-DNA Binding Sites Prediction via Integrating Pretrained Large Language Models and Contrastive Learning
This repository provides the implementation of the IPDLPre framework for predicting protein–DNA binding sites, with additional support for protein–RNA binding site prediction.  
IPDLPre is built upon three large-scale, pre-trained protein language models: ESM-2, ProtT5, and ProtBert. These models are integrated using Hugging Face's Transformers and PyTorch.  
Please ensure that all required dependencies are installed before running the code.   
ESM-2: https://huggingface.co/facebook/esm2_t12_650M_UR50D  
ProtT5: https://huggingface.co/Rostlab/prot_t5_xl_uniref50  
ProtBert: https://huggingface.co/Rostlab/prot_bert  

# 1. System Requirements  
The source code developed in Python 3.9 using PyTorch 2.5.1. The required python dependencies are given below.  
Python 3.9+  
PyTorch 2.5.1  
Torchvision 0.20.1  
Torchaudio 2.5.1  
CUDA 11.8 (recommended)  
Transformers 4.46.3  
SentencePiece 0.2.0  
fair-esm 2.0.0  
scikit-learn 1.5.2  
pandas 2.2.3  
matplotlib 3.9.4  
pytorch-lightning 1.9.5  

# 2. Feature Generation  
step1： Please download the required pretrained protein language models and place them in the specified directories as follows:
**ESM-2**  
Download from: https://huggingface.co/facebook/esm2_t33_650M_UR50D  
Target path: `./process_feature/pretrained_model/facebook/esm2_t12_650M_UR50D`

**ProtT5**  
Download from: https://huggingface.co/Rostlab/prot_t5_xl_uniref50    
Target path: `./process_feature/pretrained_model/Rostlab/prot_t5_xl_uniref50`

**ProtBert**  
Download from: https://huggingface.co/Rostlab/prot_bert    
Target path: `./process_feature/pretrained_model/Rostlab/prot_bert`

You may use the Hugging Face `transformers` library to load and cache these models automatically, or download them manually and place them as shown above.


step2：  

# 3. How to use

## 3.2 Train and test

### 3.2.1 Extract features

### 3.2.2 Train and test

## 3.3 Only For nucleotide binding residues prediction purpose

### Generate Features

After completing the steps above, run the following command to generate features and save them in the ./feature directory:

python ./process_feature/process_feature.py
Alternatively, you can download the features directly from our cloud drive: Google Drive Link.
