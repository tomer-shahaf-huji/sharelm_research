import os
import time
import polars as pl
import torch
from sentence_transformers import SentenceTransformer
import pickle



import os
import psutil

def log_mem(msg):
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"[MEM] {msg}: {rss:.2f} GB", flush=True)



ENCODER_BATCH_SIZE = 64
MAX_SEQ_LENGTH = 512


# --- Configuration ---
MODEL_NAME = "intfloat/e5-base-v2"

CACHE_ROOT = "/cs/labs/oabend/tomer.shahaf/hf_cache_root"
final_conversations_df_parquet_path = os.path.join(CACHE_ROOT, "final_conversations_df_parquet.pqt")


def read_conversations_df():
    return pl.read_parquet(final_conversations_df_parquet_path)

    
def load_model(model_name):
    t0 = time.time()
    print("Loading model...")
    model = SentenceTransformer(model_name)
    model.max_seq_length = MAX_SEQ_LENGTH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.half()  # FP16
    print(f"Model loaded on {device} in {time.time() - t0:.2f}s")
    return model

def extract_and_save_embeddings(text_rows, model, prefix):
    """
    Encodes ALL rows in one go. SentenceTransformer handles the batching.
    """
    flat_texts = []
    row_lengths = []
    
    # 1. Flatten all rows (Fast in Python for 100k items)
    for row in text_rows:
        if row is None:
            row = []
        processed_row = [f"{prefix}{p}" for p in row]
        flat_texts.extend(processed_row)
        row_lengths.append(len(processed_row))
    
    if not flat_texts:
        return [[] for _ in row_lengths]

    print(f"Encoding {len(flat_texts)} sentences with prefix '{prefix}'...")
    
    # 2. Encode Everything
    # The library handles queueing batches to the GPU automatically
    all_embeddings = model.encode(
        flat_texts, 
        batch_size=ENCODER_BATCH_SIZE, 
        normalize_embeddings=True, 
        convert_to_tensor=True,
        show_progress_bar=True 
    )
    
    # 3. Reconstruct
    # Slicing the large tensor back into rows
    first_similarity_scores = []
    sequential_similarity_scores = []
    current_idx = 0
    
    all_embeddings_by_row = []
    for idx, length in enumerate(row_lengths):
        if length > 0:
            row_embeddings = all_embeddings[current_idx : current_idx + length]
            all_embeddings_by_row.append(row_embeddings)
            current_idx += length
            
        if idx % 1_000 == 0:
          log_mem(f"after idx {idx}")
            
    
    log_mem("before pickle")
    
    p = os.path.join(CACHE_ROOT, prefix + "all_embeddings.pqt")
    with open(p, 'wb') as f:
        pickle.dump(all_embeddings_by_row, f) 
        
    log_mem("after pickle")

    # return all_embeddings_by_row
           


def process_dataset_full(conversations_df, model):
    t0 = time.time()

    # Process User Prompts
    extract_and_save_embeddings(
        conversations_df["user_prompts"].to_list(), 
        model=model,
        prefix="query: "
    )
    
    # Process Model Answers
    extract_and_save_embeddings(
        conversations_df["model_answers"].to_list(), 
        model=model,
        prefix="passage: "
    )
    

if __name__ == "__main__":
    model = load_model(MODEL_NAME)
    conversations_df = read_conversations_df()
    print(conversations_df.shape)
    process_dataset_full(conversations_df, model)
