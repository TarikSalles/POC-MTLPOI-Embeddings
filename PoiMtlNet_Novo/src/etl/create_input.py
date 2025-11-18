from configs.model import InputsConfig
from configs.paths import IoPaths, EmbeddingEngine

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List, Union, Optional
from tqdm import tqdm  # For progress tracking
import gc
from functools import lru_cache

# Define constants instead of magic numbers
SEQUENCE_LENGTH = 10
PADDING_VALUE = -1
MIN_SEQUENCE_LENGTH = 5




def parse_and_sort_checkins(checkin_timestamps: List[str]) -> List[datetime]:
    """
    Parse and sort checkin timestamps using vectorized operations.

    Args:
        checkin_timestamps: List of timestamp strings

    Returns:
        List of sorted datetime objects
    """
    return sorted(pd.to_datetime(checkin_timestamps))


def previus_generate_sequences(places_visited: List[int], max_sequences_per_user: Optional[int] = None) -> Union[
    List[List[int]], int]:
    """
    Generate fixed-length sequences from a list of places visited.

    Args:
        places_visited: List of place IDs representing sequential visits
        max_sequences_per_user: Maximum number of sequences to generate per user

    Returns:
        List of fixed-length sequences or 0 if input is too short
    """
    if len(places_visited) < MIN_SEQUENCE_LENGTH:
        return 0

    sequences = []
    for i in range(0, len(places_visited), InputsConfig.EMBEDDING_DIM):
        if len(places_visited) >= i + SEQUENCE_LENGTH:
            # Full sequence
            seq = places_visited[i:i + SEQUENCE_LENGTH]
        else:
            # Pad the last partial sequence
            remaining = places_visited[i:]
            seq = [PADDING_VALUE] * (SEQUENCE_LENGTH - len(remaining)) + remaining

        sequences.append(seq)

        # Limit sequences if max_sequences_per_user is specified
        if max_sequences_per_user and len(sequences) >= max_sequences_per_user:
            break

    return sequences


def generate_sequences(
        places_visited: List[int],
        window_size: int = InputsConfig.SLIDE_WINDOW,
        pad_value: int = PADDING_VALUE,
) -> List[List[int]]:
    """
    Generate non-overlapping sequences of fixed length where:
      - The first (window_size) positions are historical visits (padded at front if needed).
      - The final position is the next place visited, or repeats the last non-padding history entry if no next visit.

    Sequences that consist entirely of padding (including the target) are omitted.

    Args:
        places_visited: List of place IDs representing sequential visits.
        window_size: Total length of each sequence (including the target).
        pad_value: Value used for padding when there is insufficient data.

    Returns:
        A list of sequences, each of length window_size. If places_visited is empty,
        returns an empty list.
    """
    if not places_visited or len(places_visited) < MIN_SEQUENCE_LENGTH:
        return []

    sequences: List[List[int]] = []
    step = window_size
    n = len(places_visited)

    for i in range(0, n, step):
        # Build history window and pad at end if too short
        history = places_visited[i: i + step]
        if len(history) < step:
            history = history + [pad_value] * (step - len(history))

        # Determine next-place target (or use last real visit and pad that slot)
        next_idx = i + step
        if next_idx < n:
            target = places_visited[next_idx]
        else:
            # Find last non-pad value from history
            for j in range(len(history) - 1, -1, -1):
                if history[j] != pad_value:
                    target = history[j]
                    # remove that element and shift left, then pad at end
                    history = history[:j] + history[j + 1:]
                    history = history + [pad_value] * (step - len(history))
                    break
            else:
                target = pad_value  # All pad case fallback

        # Skip sequences that are all padding
        if all(x == pad_value for x in history) or target == pad_value:
            continue

        sequences.append(history + [target])

    return sequences


def generate_sequences_dataframe(users_ids: List[int], sequences: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Generate a DataFrame of sequences and save to CSV more efficiently.

    Args:
        users_ids: List of user IDs
        sequences: DataFrame containing sequences indexed by user ID
        output_path: Path to save the resulting CSV file

    Returns:
        DataFrame containing generated sequences
    """
    # Preallocate data structure
    data = []

    for user_id in tqdm(users_ids, desc="Processing users"):
        try:
            if user_id not in sequences.index:
                continue

            user_sequences = sequences.loc[user_id].iloc[0]
            if not isinstance(user_sequences, list):
                continue

            # Process all trajectories at once
            for trajectory in user_sequences:
                data.append([user_id] + trajectory)
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not process user ID {user_id}: {str(e)}")
            continue

    # Create DataFrame in one operation instead of row-by-row
    column_names = ['userid'] + list(range(1, SEQUENCE_LENGTH + 1))
    nextpoi_sequences = pd.DataFrame(data, columns=column_names)

    print(f'nextpoi_sequences shape: {nextpoi_sequences.shape}')

    # Save to CSV with proper error handling
    try:
        nextpoi_sequences.to_parquet(output_path, index=False)
        print(f'Success: nextpoi_sequences saved at {output_path}\n')
    except IOError as e:
        print(f"Error saving to {output_path}: {str(e)}")

    return nextpoi_sequences


def processing_sequences_next(df_nextpoi_sequences: pd.DataFrame,
                              output_path: str,
                              embeddings_with_category: pd.DataFrame,
                              embeddings_without_category: pd.DataFrame,
                              save_step: int = 10000) -> pd.DataFrame:
    """
    Generate input data for next POI prediction model using batch processing,
    accumulating batches in memory and writing once to Parquet at the end.

    Args:
        df_nextpoi_sequences: DataFrame containing sequences indexed by user ID
        output_path: Path to save the resulting Parquet file
        embeddings_with_category: DataFrame with a 'category' column, indexed by POI ID
        embeddings_without_category: DataFrame of embeddings (no category), indexed by POI ID
        save_step: Number of sequences to process between progress updates

    Returns:
        DataFrame of generated input data
    """
    EMB_DIM = InputsConfig.EMBEDDING_DIM
    SLID = InputsConfig.SLIDE_WINDOW

    # Build column names: 0,1,...,EMB_DIM*SLID-1, then 'next_category', 'userid'
    col_count = EMB_DIM * SLID
    column_names = [str(i) for i in range(col_count)] + ['next_category', 'userid']

    # Check for existing file to support resume functionality
    existing_data = []
    start_idx = 0
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_parquet(output_path)
            start_idx = len(existing_df)
            existing_data.append(existing_df)
            print(f"Resuming from row {start_idx} (found existing file with {start_idx} rows).")
        except Exception as e:
            print(f"Warning: Could not read existing file ({e}), starting from scratch.")
            start_idx = 0
    else:
        print("No existing file found; starting from scratch.")

    # Prepare lookups
    category_lookup = embeddings_with_category['category'].to_dict()
    emb_lookup = {
        poi: vec.values.astype(np.float32)
        for poi, vec in embeddings_without_category.iterrows()
    }
    zero_emb = np.zeros(EMB_DIM, dtype=np.float32)

    @lru_cache(maxsize=None)
    def get_embedding(poi_id):
        if poi_id == PADDING_VALUE:
            return zero_emb
        return emb_lookup.get(poi_id, zero_emb)

    # Accumulate batch DataFrames in memory
    batch_dfs = []
    total = len(df_nextpoi_sequences)
    iterator = tqdm(range(start_idx, total, save_step), desc="Processing batches")

    for batch_start in iterator:
        batch_end = min(batch_start + save_step, total)
        batch = df_nextpoi_sequences.iloc[batch_start:batch_end]

        rows_to_write = []
        for _, row in batch.iterrows():
            seq = row.iloc[:SLID].values
            target_poi = row.iloc[SLID]
            # lookup next category
            next_cat = category_lookup.get(target_poi, 'None')
            # build flattened embedding
            emb_matrix = np.vstack([get_embedding(int(p)) for p in seq])
            flat = emb_matrix.ravel()  # shape (SLID*EMB_DIM,)
            # assemble full row
            rows_to_write.append(
                list(flat) + [next_cat, str(row.name)]
            )

        # Create DataFrame for this batch and add to list
        batch_df = pd.DataFrame(rows_to_write, columns=column_names)
        batch_dfs.append(batch_df)

        iterator.set_postfix(rows_processed=batch_end)

    # Concatenate all batches at once (much more efficient than repeated concat)
    print("Concatenating all batches...")
    if existing_data:
        all_dfs = existing_data + batch_dfs
    else:
        all_dfs = batch_dfs

    final_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame(columns=column_names)

    # Write to Parquet once
    print(f"Writing {len(final_df)} sequences to {output_path}...")
    final_df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"Success: Wrote {len(final_df)} sequences to {output_path}")

    # Clean up and return
    gc.collect()
    return final_df


def generate_next_input(df_embb, df_filter,
                        sequences_path: str,
                        next_input_path: str,
                        max_sequences_per_user: Optional[int] = None) -> pd.DataFrame:
    """
    Generate input data for next POI prediction model with improved performance.

    Args:
        embeddings: DataFrame containing POI embeddings
        paths: Either a PathConfig object or a list of file paths:
               [unused, checkins_path, sequences_path, output_path]
        max_sequences_per_user: Maximum number of sequences to generate per user

    Returns:
        DataFrame of generated input data
    """

    try:
        # Load check-ins data and convert datetime in one operation
        df_filter['local_datetime'] = pd.to_datetime(df_filter['local_datetime'])
    except Exception as e:
        print(f"Error loading check-ins data: {str(e)}")
        return pd.DataFrame()

    # Process users and generate sequences
    print("Generating sequences for each user...")

    # Sort once before grouping
    df_sorted = df_filter.sort_values(by=['userid', 'local_datetime'])

    # Filter users with sufficient check-ins and generate sequences
    min_required = InputsConfig.SLIDE_WINDOW + 1

    # Group by user and get counts
    user_groups = df_sorted.groupby('userid', sort=False)
    user_counts = user_groups.size()
    valid_users = user_counts[user_counts >= min_required].index

    # Filter and generate sequences only for valid users
    checkins_sequence_by_user = (
        df_sorted[df_sorted['userid'].isin(valid_users)]
        .groupby('userid', sort=False)['placeid']
        .apply(lambda places: generate_sequences(places.tolist()))
        .reset_index(name='visit_sequence')
    )

    # keep only users with non\-empty sequences
    checkins_sequence_by_user = checkins_sequence_by_user[
        checkins_sequence_by_user.visit_sequence.map(bool)
    ]

    sequences = checkins_sequence_by_user[['userid', 'visit_sequence']].set_index('userid')
    unique_users = sequences.index.unique()

    total_sequences = sum(len(seq) for seq in checkins_sequence_by_user['visit_sequence'])
    print(f'Users with valid sequences: {len(checkins_sequence_by_user)}, Total visit sequences: {total_sequences}')

    # Prepare embeddings - optimize by creating copies only when needed
    print("Preparing embeddings...")
    embeddings_with_category = df_embb.copy()
    embeddings_without_category = df_embb.drop(columns=['category'])

    # Add padding embedding
    padding_embedding = [0] * InputsConfig.EMBEDDING_DIM

    if PADDING_VALUE not in embeddings_with_category.index:
        embeddings_with_category.loc[PADDING_VALUE] = padding_embedding + [0]
        embeddings_with_category.loc[PADDING_VALUE, 'category'] = 'None'

        padding_df = pd.DataFrame([padding_embedding],
                                  columns=embeddings_without_category.columns,
                                  index=[PADDING_VALUE])
        embeddings_without_category = pd.concat([embeddings_without_category, padding_df])

    # Generate sequences CSV
    print("Generating sequences DataFrame...")
    next_sequences = generate_sequences_dataframe(unique_users, sequences, sequences_path)
    next_sequences.set_index('userid', inplace=True)

    # Generate input data directly without re-reading the CSV
    print("Generating next POI input data...")
    next_input = processing_sequences_next(
        next_sequences,
        next_input_path,
        embeddings_with_category,
        embeddings_without_category
    )

    print("Processing complete!")
    return next_input


def generate_category_input(df_embb: pd.DataFrame, category_path: str):
    df_embb.set_index('placeid', inplace=True)
    df_embb.to_parquet(category_path)


from concurrent.futures import ProcessPoolExecutor


def create_input(state, embedd_eng: EmbeddingEngine):
    print(f"Processing state: {state} with embedding engine: {embedd_eng.value}")
    # Use IoPaths methods to get state-specific paths
    df_embedd = IoPaths.load_embedd(state, embedd_eng)
    df_filter = IoPaths.load_city(state)
    sequences_path = IoPaths.get_seq_next(state, embedd_eng)
    next_input_path = IoPaths.get_next(state, embedd_eng)
    category_input_path = IoPaths.get_category(state, embedd_eng)

    sequences_path.parent.mkdir(parents=True, exist_ok=True)
    next_input_path.parent.mkdir(parents=True, exist_ok=True)
    category_input_path.parent.mkdir(parents=True, exist_ok=True)

    generate_category_input(df_embedd, str(category_input_path))
    generate_next_input(df_embedd, df_filter, str(sequences_path), str(next_input_path))


if __name__ == '__main__':
    # STATE_NAME = [("state", EmbeddingEngine.HMRM), ...]
    STATE_NAME = [
        # ("florida", EmbeddingEngine.DGI),
        ("california", EmbeddingEngine.DGI),
        ("texas", EmbeddingEngine.DGI),
        ("alabama", EmbeddingEngine.DGI),
        ("arizona", EmbeddingEngine.DGI),
        ("georgia", EmbeddingEngine.DGI),
    ]
    with ProcessPoolExecutor(max_workers=12) as executor:
        # Use submit with unpacking for each state
        futures = [executor.submit(create_input, state, engine) for state, engine in STATE_NAME]
        # Wait for all to complete
        results = [future.result() for future in futures]
