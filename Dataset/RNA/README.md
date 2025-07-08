
This directory is intended to store the extracted protein language model (PLM) embeddings for RNA-related samples.  
These embeddings are **not included in the repository** due to their large file size.

---

## 🔧 How to Generate the Embeddings

You can generate the required PLM embeddings by running the following Python scripts:

- `pre_esm2.py`  → generates ESM-2 embeddings  
- `pre_protT5.py` → generates ProtT5 embeddings  
- `pre_protbert.py` → generates ProtBert embeddings  

These scripts will automatically process your data and store the resulting embeddings in this directory:
