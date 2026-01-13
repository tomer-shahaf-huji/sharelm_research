import os
import time
import polars as pl
import torch
from sentence_transformers import SentenceTransformer


import psutil

def log_mem(msg):
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"[MEM] {msg}: {rss:.2f} GB", flush=True)

ENCODER_BATCH_SIZE = 256
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
    log_mem("before reconstruction")
    # Move to CPU and numpy
    all_embeddings_np = all_embeddings.cpu().to(torch.float32).numpy()
    
    embeddings_list = []
    current_idx = 0
    
    # Reconstruct rows
    for i, length in enumerate(row_lengths):
        if length > 0:
            # formatting: each row is a list of embeddings (which are lists of floats)
            row_emb = all_embeddings_np[current_idx : current_idx + length].tolist()
            embeddings_list.append(row_emb)
            current_idx += length
        else:
            embeddings_list.append([])
            
        if i % 10_000 == 0 and i > 0:
            log_mem(f"processed {i} rows")

    log_mem("after reconstruction")
    
    # 4. Save as Parquet
    df = pl.DataFrame({"embeddings": embeddings_list})
    save_embeddings_to_parquet(df, prefix)
        
    log_mem("after save")


def save_embeddings_to_parquet(df, prefix):
    # Sanitize prefix for filename
    # Why safe_prefix is needed:
    # 1. 'query: ' contains a colon and a space.
    # 2. Colons are reserved characters in some filesystems (Windows) and confusing in others (Linux).
    # 3. Spaces require escaping in the shell (e.g. "query\ all_embeddings.pqt").
    # 4. Replacing them with underscores ensures a clean, portable filename like "query_all_embeddings.pqt".
    safe_prefix = prefix.replace(": ", "_").replace(":", "_").replace(" ", "_")
    
    filename = f"{safe_prefix}all_embeddings.pqt"
    p = os.path.join(CACHE_ROOT, filename)
    
    print(f"Saving dataframe shape {df.shape} to {p}...", flush=True)
    df.write_parquet(p)
        
    log_mem("after save")

        

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
