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

# 2. Feature Generation  
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
Each script processes protein sequences and saves the corresponding embeddings into the ./Dataset/ directory under subfolders such as ./Dataset/DNA/, ./Dataset/RNA/, or ./Dataset/DRNA/, depending on the type of data being processed.  
After running these scripts, you will obtain .pkl files that contain the extracted embeddings. For example:  
DNA-related embeddings will be saved in ./Dataset/DNA/  
RNA-related embeddings will be saved in ./Dataset/RNA/  
DRNA-related embeddings will be saved in ./Dataset/DRNA/  
These .pkl files are used as input for downstream model training and evaluation.  

**Example usage:**
```bash
python pre_esm2.py
python pre_protT5.py
python pre_protbert.py
```
⚠️ If you are unable to generate these embeddings locally due to hardware or runtime limitations, pre-extracted .pkl files can be downloaded from our cloud drive:  
📦 Google Drive – Single Model Embeddings  
(Includes ProtT5, ProtBert, and ESM-2 features for DNA/RNA/DRNA)  
Please place the downloaded files into the appropriate subfolders under ./Dataset/.  

## step3 ：Concatenate Multi-Model Embeddings  
Once the individual embeddings from ESM-2, ProtT5, and ProtBert have been generated (see **step2**), you can run the script below to concatenate them into a unified embedding representation for each protein sequence:  
```bash
python merge_embeddings.py
```
The script will output concatenated .pkl files such as:  
./Dataset/DNA/three_DNA_Train.pkl    
./Dataset/RNA/three_RNA_Train.pkl  
./Dataset/DRNA/three_DRNA_Train.pkl  
These files serve as the final input to the downstream predictive model.   
🔄 Alternatively, if you wish to skip local generation, you can download the concatenated feature files directly from our cloud drive:  
📦 [Google Drive – DNA/RNA/DRNA_embedding (e.g., three_DNA_Train.pkl, three_RNA_Train.pkl, etc.)](https://drive.google.com/drive/folders/1fEPL1xJZbGAo6qmFj-cxdiG3HtibIcVu)  

📁 Be sure to place the files in the correct locations:  
./Dataset/DNA/three_DNA_Train.pkl   
./Dataset/DNA/three_DNA_Test.pkl  

# 3. Run IPDLPre for prediction
To run prediction using the IPDLPre framework on datasets located in the ./Dataset directory, execute the following command:
```bash
python main.py
```
Before running the script, please ensure that:  
All dependencies are properly installed (see Requirements).  
The .pkl feature files (generated or downloaded) are placed in the correct subdirectories under ./Dataset/, such as ./Dataset/DNA/, ./Dataset/RNA/, or ./Dataset/DRNA/.  
Configuration settings (e.g., dataset name, paths, model parameters) in the code are correctly set according to your usage.  


