import os
import datasets
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import time

# --- Configuration ---
MODEL_NAME = "intfloat/e5-base-v2"
DATASET_PATH = "ours_dataset_medium_conversations.pqt"

# 1. High Map Batch: Reduce Python loop overhead
MAP_BATCH_SIZE = 100  

# 2. Strict Encoder Batch: Maximize GPU without OOM
ENCODER_BATCH_SIZE = 64 

# E5-small/base supports up to 512 tokens
MAX_SEQ_LENGTH = 512 

def load_model(model_name):
    t0 = time.time()
    print("loading model")
    model = SentenceTransformer(model_name)
    model.max_seq_length = MAX_SEQ_LENGTH
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.half() # FP16 for speed/memory
    print(f"Model loaded on {device} with max_seq_length={model.max_seq_length} in {time.time()-t0} seconds")
    return model

def load_dataset(dataset_path):
    t0 = time.time()
    df = pd.read_parquet(dataset_path)
    dataset = datasets.Dataset.from_pandas(df)
    #vdataset = datasets.load_dataset("parquet", data_files=dataset_path, split="train")
    print(f"Loaded dataset with shape: {dataset.shape} in {time.time()-t0} seconds")
    return dataset

def _get_embeddings_and_cosines(text_rows, prefix):
    """
    Helper function to process a list of rows, encode them with a specific prefix,
    and calculate cosine similarity to the first item in each row.
    """
    flat_texts = []
    row_lengths = []
    
    # 1. Flatten and Pre-process
    for row in text_rows:
        if row is None:
            row = []
            
        # Apply the specific prefix (query vs passage)
        processed_row = [f"{prefix}{p}" for p in row]
        flat_texts.extend(processed_row)
        row_lengths.append(len(processed_row))
    
    # 2. Encode
    all_embeddings = model.encode(
        flat_texts, 
        batch_size=ENCODER_BATCH_SIZE, 
        normalize_embeddings=True, 
        convert_to_tensor=True,
        show_progress_bar=False 
    )
    
    # 3. Reconstruct and Calculate
    similarity_scores = []
    current_idx = 0
    
    for length in row_lengths:
        row_embeddings = all_embeddings[current_idx : current_idx + length]
        
        if length > 0:
            anchor_vector = row_embeddings[0]
            scores = torch.matmul(row_embeddings, anchor_vector)
            similarity_scores.append(scores.tolist())
        else:
            similarity_scores.append([])
            
        current_idx += length
        
    return similarity_scores

def extract_semantic_vectors_cosines_batch(batch):
    # Process User Prompts -> Use "query: "
    user_cosines = _get_embeddings_and_cosines(
        batch["user_prompts"], 
        prefix="query: "
    )
    
    # Process Model Answers -> Use "passage: "
    model_cosines = _get_embeddings_and_cosines(
        batch["model_answers"], 
        prefix="passage: "
    )
    
    return {
        "user_prompts_similarity": user_cosines,
        "model_answers_similarity": model_cosines
    }

def process_dataset(dataset):
    t0 = time.time()
    print(f"Starting map with Map Batch: {MAP_BATCH_SIZE} | Encoder Batch: {ENCODER_BATCH_SIZE}")
    updated_dataset = dataset.map(
        extract_semantic_vectors_cosines_batch,
        batched=True,
        batch_size=MAP_BATCH_SIZE, 
        desc="Extracting cosines for prompts and answers" 
    )
    
    output_filename = "ours_dataset_medium_conversations_with_cosines.pqt"
    updated_dataset.to_parquet(output_filename)
    print(f"Saved to {output_filename} in {time.time()-t0} seconds")


if __name__ == "__main__":
    model = load_model(MODEL_NAME)
    ds = load_dataset(DATASET_PATH)
    process_dataset(ds)
