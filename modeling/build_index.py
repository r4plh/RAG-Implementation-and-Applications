from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from datasets import Dataset
# from utils.config import MODEL_PATH, DATA_PATH, DEVICE
import faiss
import os
import torch

DATA_PATH = "data_files"
MODEL_PATH = "model_checkpoints"
DEVICE = "cuda"

def build_index():
    tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(DEVICE)
    model.eval()

    docs = Dataset.load_from_disk(os.path.join(DATA_PATH, 'train_dataset'))['passages']['text']
    docs = list(set([d for sublist in docs for d in sublist]))[:10000]  # Flatten and deduplicate
    embeddings = []

    for doc in docs:
        inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            embeddings.append(model(**inputs).pooler_output.cpu().numpy())

    embeddings = torch.cat(embeddings)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings.numpy())

    faiss.write_index(faiss_index, os.path.join(MODEL_PATH, "faiss_index.bin"))

if __name__ == "__main__":
    build_index()
