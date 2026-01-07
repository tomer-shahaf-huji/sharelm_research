import os
import time
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

# --- Configuration ---
MODEL_NAME = "intfloat/e5-base-v2"

CACHE_ROOT = "/cs/labs/oabend/tomer.shahaf/hf_cache_root"
INPUT_CHUNKS_DIR = os.path.join(CACHE_ROOT, "processed_chunks")
OUTPUT_CHUNKS_DIR = os.path.join(CACHE_ROOT, "processed_chunks_with_cosines")

ENCODER_BATCH_SIZE = 64
MAX_SEQ_LENGTH = 512


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


def _get_embeddings_and_cosines(text_rows, model, prefix):
    flat_texts = []
    row_lengths = []

    for row in text_rows:
        if row is None:
            row = []
        processed_row = [f"{prefix}{p}" for p in row]
        flat_texts.extend(processed_row)
        row_lengths.append(len(processed_row))

    if not flat_texts:
        return [[] for _ in row_lengths], [[] for _ in row_lengths]

    print(f"Encoding {len(flat_texts)} sentences ({prefix.strip()})")

    all_embeddings = model.encode(
        flat_texts,
        batch_size=ENCODER_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    first_similarity_scores = []
    sequential_similarity_scores = []
    current_idx = 0

    for length in row_lengths:
        if length > 0:
            row_embeddings = all_embeddings[current_idx : current_idx + length]

            # Similarity to first
            anchor = row_embeddings[0]
            sims_first = torch.matmul(row_embeddings, anchor)
            first_similarity_scores.append(sims_first.tolist())

            # Sequential similarity
            if length > 1:
                curr = row_embeddings[1:]
                prev = row_embeddings[:-1]
                seq_sims = (curr * prev).sum(dim=1)
                first_one = torch.tensor(
                    [1.0], device=row_embeddings.device, dtype=row_embeddings.dtype
                )
                seq_sims = torch.cat([first_one, seq_sims])
            else:
                seq_sims = torch.tensor(
                    [1.0], device=row_embeddings.device, dtype=row_embeddings.dtype
                )

            sequential_similarity_scores.append(seq_sims.tolist())
        else:
            first_similarity_scores.append([])
            sequential_similarity_scores.append([])

        current_idx += length

    return first_similarity_scores, sequential_similarity_scores


def process_single_chunk(chunk_path, model, output_dir):
    t0 = time.time()
    fname = os.path.basename(chunk_path)
    print(f"\n=== Processing {fname} ===")

    df = pl.read_parquet(chunk_path)

    user_first, user_seq = _get_embeddings_and_cosines(
        df["user_prompts"].to_list(),
        model=model,
        prefix="query: ",
    )

    model_first, model_seq = _get_embeddings_and_cosines(
        df["model_answers"].to_list(),
        model=model,
        prefix="passage: ",
    )

    updated_df = df.with_columns(
        [
            pl.Series("user_prompts_similarity_to_first", user_first),
            pl.Series("user_prompts_sequential_similarity", user_seq),
            pl.Series("model_answers_similarity_to_first", model_first),
            pl.Series("model_answers_sequential_similarity", model_seq),
        ]
    )

    output_path = os.path.join(
        output_dir, fname.replace(".pqt", "_with_cosines.pqt")
    )
    updated_df.write_parquet(output_path)

    print(f"Saved → {output_path} ({time.time() - t0:.2f}s)")


def process_all_chunks(input_dir, output_dir, model):
    os.makedirs(output_dir, exist_ok=True)

    chunk_files = sorted(
        f for f in os.listdir(input_dir) if f.endswith(".pqt")
    )

    print(f"Found {len(chunk_files)} chunks")

    for fname in chunk_files:
        chunk_path = os.path.join(input_dir, fname)
        process_single_chunk(chunk_path, model, output_dir)


if __name__ == "__main__":
    model = load_model(MODEL_NAME)
    process_all_chunks(INPUT_CHUNKS_DIR, OUTPUT_CHUNKS_DIR, model)
