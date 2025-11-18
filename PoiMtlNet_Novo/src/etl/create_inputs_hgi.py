from src.configs.model import InputsConfig
from src.configs.paths import OUTPUT_DIR
import csv
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List, Union, Dict, Optional, Any
from tqdm import tqdm  # For progress tracking
import gc
from functools import lru_cache

# Define constants instead of magic numbers
SEQUENCE_LENGTH = 10
PADDING_VALUE = -1
MIN_SEQUENCE_LENGTH = 5


class PathConfig:
    """Class to handle file paths configuration."""

    def __init__(self, base_dir: str, checkins_file: str, sequences_file: str, output_file: str):
        self.base_dir = base_dir
        self.checkins_path = os.path.join(base_dir, checkins_file)
        self.sequences_path = os.path.join(base_dir, sequences_file)
        self.output_path = os.path.join(base_dir, output_file)

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.base_dir, exist_ok=True)

    def as_list(self) -> List[str]:
        """Return paths as a list for backward compatibility."""
        return ["", self.checkins_path, self.sequences_path, self.output_path]


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
        nextpoi_sequences.to_csv(output_path, index=False)
        print(f'Success: nextpoi_sequences saved at {output_path}\n')
    except IOError as e:
        print(f"Error saving to {output_path}: {str(e)}")

    return nextpoi_sequences


def processing_sequences_next(df_nextpoi_sequences: pd.DataFrame,
                              output_path: str,
                              embeddings_with_category: pd.DataFrame,
                              embeddings_without_category: pd.DataFrame,
                              save_step: int = 10000) -> pd.DataFrame:

    print(output_path)
    """
    Generate input data for next POI prediction model using batch processing,
    writing incrementally to CSV for much faster performance.

    Args:
        df_nextpoi_sequences: DataFrame containing sequences indexed by user ID
        output_path: Path to save the resulting CSV file
        embeddings_with_category: DataFrame with a 'category' column, indexed by POI ID
        embeddings_without_category: DataFrame of embeddings (no category), indexed by POI ID
        save_step: Number of sequences to process between CSV flushes

    Returns:
        DataFrame of generated input data (reads entire CSV at end for return)
    """
    EMB_DIM = InputsConfig.EMBEDDING_DIM
    SLID = InputsConfig.SLIDE_WINDOW

    # Build column names: 0,1,...,EMB_DIM*SLID-1, then 'next_category', 'userid'
    col_count = EMB_DIM * SLID
    column_names = [str(i) for i in range(col_count)] + ['next_category', 'userid']

    # Determine where to start
    if os.path.exists(output_path):
        # Count existing rows (minus header)
        with open(output_path, 'r', newline='') as f:
            existing = sum(1 for _ in f) - 1
        start_idx = max(existing, 0)
        write_header = False
        print(f"Resuming from row {start_idx} (found existing file).")
    else:
        start_idx = 0
        write_header = True
        print("No existing file found; starting from scratch.")

    # Prepare lookups
    category_lookup = embeddings_with_category['category'].to_dict()
    print("Exemplo de category_lookup:")
    sample_items = list(category_lookup.items())[:5]
    print(sample_items)

    emb_lookup = {
        poi: vec.values.astype(np.float32)
        for poi, vec in embeddings_without_category.iterrows()
    }
    zero_emb = np.zeros(EMB_DIM, dtype=np.float32)

    @lru_cache(maxsize=None)
    def get_embedding(poi_id: str):
        if poi_id == str(PADDING_VALUE):
            return zero_emb
        return emb_lookup.get(str(poi_id), zero_emb)

    # Open CSV for append (or write+header)
    csv_file = open(output_path, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(column_names)

    total = len(df_nextpoi_sequences)
    iterator = tqdm(range(start_idx, total, save_step), desc="Batches")
    for batch_start in iterator:
        batch_end = min(batch_start + save_step, total)
        batch = df_nextpoi_sequences.iloc[batch_start:batch_end]

        rows_to_write = []
        for _, row in batch.iterrows():
            seq = row.iloc[:SLID].values
            target_poi = row.iloc[SLID]

            # sempre usar string como chave
            target_poi_str = str(target_poi)

            # lookup next category
            next_cat = category_lookup.get(target_poi_str, 'None')

            # build flattened embedding
            emb_matrix = np.vstack([get_embedding(str(p)) for p in seq])
            flat = emb_matrix.ravel()  # shape (SLID*EMB_DIM,)

            rows_to_write.append(
                list(flat) + [next_cat, str(row.name)]
            )



        # write batch
        writer.writerows(rows_to_write)
        csv_file.flush()
        gc.collect()
        iterator.set_postfix(rows_written=batch_end)

    csv_file.close()
    print(f"Finished: wrote {total} sequences to {output_path}")

    # Return full DataFrame for downstream use
    return output_path


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
        def ensure_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                return df
            if 'local_datetime' in df.columns:
                df = df.rename(columns={'local_datetime': 'datetime'})
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                return df
            raise KeyError("Esperava coluna 'datetime' ou 'local_datetime'.")

        df_filter = ensure_datetime_column(df_filter)
    except Exception as e:
        print(f"Error loading check-ins data: {str(e)}")
        return pd.DataFrame()

    # Process users and generate sequences
    print("Generating sequences for each user...")

    # sort once, group, apply and filter in a vectorized way
    checkins_sequence_by_user = (
        df_filter
        .sort_values(by=['userid', 'datetime'])
        .groupby('userid')['placeid']
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

    # 1) garantir índice por placeid e separar category
    if 'placeid' in df_embb.columns:
        embeddings_with_category = df_embb.set_index('placeid').copy()
    else:
        # já pode estar indexado; garantir que 'category' exista
        embeddings_with_category = df_embb.copy()

    if 'category' not in embeddings_with_category.columns:
        embeddings_with_category['category'] = 'None'
    embeddings_with_category.index = embeddings_with_category.index.astype(str)


    # 2) detectar automaticamente as colunas numéricas dos embeddings
    num_cols = [c for c in embeddings_with_category.columns if str(c).isdigit()]
    EMB_DIM = len(num_cols)            # aqui pode ser 106
    SLID = InputsConfig.SLIDE_WINDOW   # permanece como está

    # 3) construir os dois DFs alinhados
    embeddings_without_category = embeddings_with_category[num_cols].copy()

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
    df_embb.to_csv(category_path)


from concurrent.futures import ProcessPoolExecutor


from concurrent.futures import ProcessPoolExecutor


from sklearn.decomposition import PCA
def process_state(state):
    print(f"[START] Processando estado: {state}")

    # Embeddings puros
    df_loc_src = pd.read_csv(f"{OUTPUT_DIR}/{state}/location_embedding.csv")      # loc (128D)
    df_poi     = pd.read_csv(f"{OUTPUT_DIR}/{state}/embeddings.csv")  # poi (128D)
    time_emb   = pd.read_csv(f"{OUTPUT_DIR}/{state}/time_embedding.csv")          # time (128D)

    # Garante mesmo tamanho entre as três fontes
    min_len = min(len(df_loc_src), len(df_poi), len(time_emb))
    if len(df_loc_src) != min_len:
        df_loc_src = df_loc_src.iloc[:min_len].reset_index(drop=True)
    if len(df_poi) != min_len:
        df_poi = df_poi.iloc[:min_len].reset_index(drop=True)
    if len(time_emb) != min_len:
        time_emb = time_emb.iloc[:min_len].reset_index(drop=True)

    # placeid como string
    if "placeid" in df_loc_src.columns:
        df_loc_src["placeid"] = df_loc_src["placeid"].astype(str)
    if "placeid" in df_poi.columns:
        df_poi["placeid"] = df_poi["placeid"].astype(str)
    if "placeid" in time_emb.columns:
        time_emb["placeid"] = time_emb["placeid"].astype(str)

    # Colunas numéricas
    num_cols_loc = sorted([c for c in df_loc_src.columns if str(c).isdigit()],
                          key=lambda x: int(x))
    num_cols_poi = sorted([c for c in df_poi.columns if str(c).isdigit()],
                          key=lambda x: int(x))
    num_cols_time = sorted([c for c in time_emb.columns if str(c).isdigit()],
                           key=lambda x: int(x))

    X_loc  = df_loc_src[num_cols_loc].astype(np.float32).to_numpy()
    X_poi  = df_poi[num_cols_poi].astype(np.float32).to_numpy()
    X_time = time_emb[num_cols_time].astype(np.float32).to_numpy()

    # Sanity checks
    if X_loc.shape[1] != X_time.shape[1]:
        raise ValueError(f"Dim loc={X_loc.shape[1]} != Dim time={X_time.shape[1]}")
    if X_poi.shape[1] != X_time.shape[1]:
        raise ValueError(f"Dim poi={X_poi.shape[1]} != Dim time={X_time.shape[1]}")

    # ================= CATEGORY =================
    # Category embedding = soma(location + time)
    X_cat = X_loc + X_time                  # (N, 128)
    CAT_DIM = X_cat.shape[1]
    cat_cols = [str(i) for i in range(CAT_DIM)]

    df_cat_emb = pd.DataFrame(X_cat, columns=cat_cols)
    df_cat_emb["placeid"] = df_loc_src["placeid"].astype(str)

    # category vem do loc (mesmo checkin)
    if "category" in df_loc_src.columns:
        df_cat_emb["category"] = df_loc_src["category"]
    else:
        df_cat_emb["category"] = "None"

    # ================= NEXT =================
    # Next embedding = soma(time + poi)
    X_next = X_poi + X_time + X_loc           # (N, 128)
    NEXT_DIM = X_next.shape[1]
    next_cols = [str(i) for i in range(NEXT_DIM)]

    df_next_emb = pd.DataFrame(X_next, columns=next_cols)
    df_next_emb["placeid"] = df_cat_emb["placeid"].astype(str)
    df_next_emb["category"] = df_cat_emb["category"]

    # Paths de saída
    output_path = f"{OUTPUT_DIR}/{state}/pre-processing/"
    os.makedirs(output_path, exist_ok=True)

    category_input_path = f"{output_path}category-input.csv"
    sequences_path      = f"{output_path}poi-sequences.csv"
    next_input_path     = f"{output_path}next-input.csv"

    # Filtro de check-ins
    df_filter = pd.read_csv(f"{OUTPUT_DIR}/{state}/filtrado.csv")

    # CATEGORY: agora usando loc+time
    generate_category_input(df_cat_emb, category_input_path)

    # NEXT: usando poi+time
    generate_next_input(df_next_emb, df_filter, sequences_path, next_input_path)

    gc.collect()
    print(f"[END] Estado {state} processado.")




if __name__ == "__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser(
        description="Gera category-input, poi-sequences e next-input para um ou mais estados"
    )
    parser.add_argument(
        "states", nargs="+", type=str,
        help="Lista de estados. Ex: montana california florida"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Número de processos em paralelo (default=4)"
    )
    args = parser.parse_args()

    states = [s.lower().strip() for s in args.states]
    print(f"[INFO] process_state para: {states}")

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        executor.map(process_state, states)
