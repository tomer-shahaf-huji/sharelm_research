import os
import polars as pl
import torch
from sentence_transformers import SentenceTransformer
import time

# --- Configuration ---
MODEL_NAME = "intfloat/e5-base-v2"
CACHE_ROOT = "/cs/labs/oabend/tomer.shahaf/hf_cache_root"
RESEARCH_DF_TMP_PARQUET_PATH = os.path.join(CACHE_ROOT, "ours_dataset_medium_conversations.pqt")
OUTPUT_PATH = os.path.join(CACHE_ROOT, "ours_dataset_medium_conversations_with_cosines.pqt")

# ONLY ONE BATCH SIZE NEEDED (For GPU VRAM)
ENCODER_BATCH_SIZE = 64 
MAX_SEQ_LENGTH = 512 

def load_model(model_name):
    t0 = time.time()
    print("Loading model...")
    model = SentenceTransformer(model_name)
    model.max_seq_length = MAX_SEQ_LENGTH
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.half() # FP16
    print(f"Model loaded on {device} in {time.time()-t0:.2f}s")
    return model

def _get_embeddings_and_cosines(text_rows, model, prefix):
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
    
    for length in row_lengths:
        if length > 0:
            row_embeddings = all_embeddings[current_idx : current_idx + length]
            
            # 1. Similarity to First (Index 0)
            anchor_vector = row_embeddings[0]
            scores_first = torch.matmul(row_embeddings, anchor_vector)
            first_similarity_scores.append(scores_first.tolist())
            
            # 2. Sequential Similarity (Index i vs i-1)
            if length > 1:
                curr = row_embeddings[1:]
                prev = row_embeddings[:-1]
                # Dot product along dim 1
                seq_sims = (curr * prev).sum(dim=1)
                # Prepend 1.0 for the first element
                first_one = torch.tensor([1.0], device=row_embeddings.device, dtype=row_embeddings.dtype)
                scores_seq = torch.cat([first_one, seq_sims])
            else:
                scores_seq = torch.tensor([1.0], device=row_embeddings.device, dtype=row_embeddings.dtype)
            
            sequential_similarity_scores.append(scores_seq.tolist())

        else:
            first_similarity_scores.append([])
            sequential_similarity_scores.append([])
            
        current_idx += length
        
    return first_similarity_scores, sequential_similarity_scores

def save_parquet(df, output_path):
    t0 = time.time()
    df.write_parquet(output_path)
    print(f"Done! Saved to {output_path} in {time.time()-t0:.2f}s")

def process_dataset_full(dataset_path, model):
    t0 = time.time()
    
    # Load entire file
    df = pl.read_parquet(dataset_path)
    print(f"Loaded {df.shape[0]} rows. Starting processing...")

    # Process User Prompts
    user_cosines_to_first, user_seq_cosines = _get_embeddings_and_cosines(
        df["user_prompts"].to_list(), 
        model=model,
        prefix="query: "
    )
    
    # Process Model Answers
    model_cosines_to_first, model_seq_cosines = _get_embeddings_and_cosines(
        df["model_answers"].to_list(), 
        model=model,
        prefix="passage: "
    )
    
    # Save
    updated_df = df.with_columns([
        pl.Series("user_prompts_similarity_to_first", user_cosines_to_first),
        pl.Series("user_prompts_sequential_similarity", user_seq_cosines),
        pl.Series("model_answers_similarity_to_first", model_cosines_to_first),
        pl.Series("model_answers_sequential_similarity", model_seq_cosines)
    ])
    
    return updated_df

if __name__ == "__main__":
    os.makedirs(CACHE_ROOT, exist_ok=True)
    model = load_model(MODEL_NAME)
    updated_df = process_dataset_full(RESEARCH_DF_TMP_PARQUET_PATH, model)
    save_parquet(updated_df, OUTPUT_PATH)