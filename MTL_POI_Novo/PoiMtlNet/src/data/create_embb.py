import os

import pandas as pd

from configs.model import InputsConfig
from configs.paths import OUTPUT_ROOT, IO_CHECKINS
from src.data.embeddings.hmrm.hmrm_new import HmrmBaselineNew


def etl_checkins(df: pd.DataFrame):
    """Clean and filter check-ins data, keeping only users with 40+ check-ins."""
    print(f'Original checkins: {df.shape}')

    # Standardize datetime column name
    if 'local_datetime' in df.columns:
        df.rename(columns={'local_datetime': 'datetime'}, inplace=True)
        print(f"Renamed 'local_datetime' to 'datetime'")


    checkins_per_user = df['userid'].value_counts()
    # The user must have at least the InputsConfig.SLIDE_WINDOW + 1 check-ins
    # because we need to predict the next check-in and the user must have at least
    # InputsConfig.SLIDE_WINDOW check-ins to be able to predict the next one
    selected_users = checkins_per_user[checkins_per_user >= InputsConfig.SLIDE_WINDOW+1]
    users_ids = selected_users.index.unique().tolist()

    print(f'Number of qualified users: {len(users_ids)}')

    filtered_checkins = df[df['userid'].isin(users_ids)]
    print(f'Filtered checkins shape: {filtered_checkins.shape}')

    return filtered_checkins


def _create_embeddings(input_file, weight=0.1, K=7, embedding_size=50):
    """Generate embeddings using HMRM with specified parameters."""
    print(f'Creating embeddings with weight={weight}, K={K}, embedding_size={embedding_size}')
    hmrm = HmrmBaselineNew(input_file, weight, K, embedding_size)
    return hmrm.start()


def create_embeddings(state_name, path, **kwargs):
    """Process checkins for a state and generate embeddings."""
    print(f'\nProcessing {state_name.capitalize()} check-ins...')

    try:
        # Create state output directory if it doesn't exist
        state_dir = f'{OUTPUT_ROOT}/{state_name}'
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)
            print(f'Created directory: {state_dir}')

        # Clean and preprocess the checkins data
        df = pd.read_csv(path, index_col=False)
        df = etl_checkins(df)

        # Save the filtered checkins
        etl_path = f'{state_dir}/filtrado.csv'
        df.to_csv(etl_path, index=False)
        print(f'Filtered check-ins saved to {etl_path}')

        # Generate embeddings
        embeddings = _create_embeddings(etl_path,
                                        kwargs.get('weight', 0.1),
                                        kwargs.get('K', 7),
                                        kwargs.get('embedding_size', 50)
                                        )

        # Save the embeddings
        embb_path = f'{state_dir}/embeddings.csv'
        embeddings.to_csv(embb_path, index=False)

        print(f'Shape: {embeddings.shape}')
        print(f'Embeddings for {state_name.capitalize()} generated successfully')

        return embeddings

    except Exception as e:
        print(f'Error processing {state_name}: {str(e)}')
        raise


if __name__ == '__main__':
    path_alabama = os.path.join(IO_CHECKINS, 'Alabama.csv')
    path_arizona = os.path.join(IO_CHECKINS, 'Arizona.csv')
    path_georgia = os.path.join(IO_CHECKINS, 'Georgia.csv')

    path_california = os.path.join(IO_CHECKINS, 'California.csv')
    path_florida = os.path.join(IO_CHECKINS, 'Florida.csv')
    path_texas = os.path.join(IO_CHECKINS, 'Texas.csv')
    path_montana = os.path.join(IO_CHECKINS, 'checkins_Montana.csv')

    # _ = create_embeddings('alabama', path_alabama, weight=0.1, K=7, embedding_size=50)
    # _ = create_embeddings('arizona', path_arizona, weight=0.1, K=7, embedding_size=50)
    # _ = create_embeddings('georgia', path_georgia, weight=0.1, K=7, embedding_size=50)
    # _ = create_embeddings('california', path_california, weight=0.1, K=7, embedding_size=50)
    _ = create_embeddings('montana', path_montana, weight=0.1, K=7, embedding_size=64)
    # _ = create_embeddings('texas', path_texas, weight=0.1, K=7, embedding_size=50)
