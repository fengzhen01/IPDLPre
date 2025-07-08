# Protein-DNA Binding Sites Prediction via Integrating Pretrained Large Language Models and Contrastive Learning
This repository provides the implementation of the IPDLPre framework for predicting protein–DNA binding sites, with additional support for protein–RNA binding site prediction.  
IPDLPre is built upon three large-scale, pre-trained protein language models: ESM-2, ProtT5, and ProtBert. These models are integrated using Hugging Face's Transformers and PyTorch.  
Please ensure that all required dependencies are installed before running the code.   

# 1.System Requirements  
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

# 2.Feature Generation  
## step1 ：Download pretrained protein language model  
Please download the required pretrained protein language models and place them in the specified directories as follows:  
**ESM-2**  
Download from: [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D)  
Target path: `./process_feature/esm2_t33_650M_UR50D`  
**ProtT5**  
Download from: [Rostlab/prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50)   
Target path: `./process_feature/prot_t5_xl_uniref50`  
**ProtBert**  
Download from: [Rostlab/prot_bert](https://huggingface.co/Rostlab/prot_bert)   
Target path: `./process_feature/prot_bert`  
You can either use the Hugging Face `transformers` library to automatically download and cache these models at runtime, or download them manually and place them in the corresponding directories as shown above.

## step2 ：Generate Protein Language Model Embeddings  
After downloading the pretrained models in **step1**, you can use the following Python scripts to generate embeddings for protein sequences:
`pre_esm2.py`  → generates embeddings using ESM-2  
`pre_protT5.py` → generates embeddings using ProtT5  
`pre_protbert.py` → generates embeddings using ProtBert
Each script processes protein sequences and saves the corresponding embeddings into the `./Dataset/` directory under subfolders such as `./Dataset/DNA/` or `./Dataset/RNA/`.

Example usage:

```bash
python pre_esm2.py
python pre_protT5.py
python pre_protbert.py
 ```` ``` ```` 

## step3 ：Generate Protein Language Model Embeddings  


# 3. How to use

## 3.2 Train and test

### 3.2.1 Extract features

### 3.2.2 Train and test

## 3.3 Only For nucleotide binding residues prediction purpose

### Generate Features

After completing the steps above, run the following command to generate features and save them in the ./feature directory:

python ./process_feature/process_feature.py
Alternatively, you can download the features directly from our cloud drive: Google Drive Link.
