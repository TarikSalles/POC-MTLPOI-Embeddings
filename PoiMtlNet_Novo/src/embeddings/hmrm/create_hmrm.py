import pandas as pd

from configs.model import InputsConfig
from configs.paths import IoPaths, IO_CHECKINS, EmbeddingEngine
from embeddings.hmrm.hmrm_new import HmrmBaselineNew


def _create_embeddings(input_file, weight=0.1, K=7, embedding_size=50) -> pd.DataFrame:
    """Generate embeddings using HMRM with specified parameters."""
    print(f'Creating embeddings with weight={weight}, K={K}, embedding_size={embedding_size}')
    hmrm = HmrmBaselineNew(input_file, weight, K, embedding_size)
    return hmrm.start()


def process_embeddings(state_name, **kwargs):
    """Process checkins for a state and generate embeddings."""
    print(f'\nProcessing {state_name.capitalize()} check-ins...')

    try:
        # Generate embeddings
        embeddings = _create_embeddings(IoPaths.get_city(state_name),
                                        kwargs.get('weight', 0.1),
                                        kwargs.get('K', 7),
                                        kwargs.get('embedding_size', 50)
                                        )

        # Save the embeddings
        embb_path = IoPaths.get_embedd(state_name, EmbeddingEngine.HMRM)
        embb_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings.to_parquet(embb_path, index=False)

        print(f'Shape: {embeddings.shape}')
        print(f'Embeddings for {state_name.capitalize()} generated successfully')

        return embeddings

    except Exception as e:
        print(f'Error processing {state_name}: {str(e)}')
        raise


if __name__ == '__main__':
    _ = process_embeddings('alabama', weight=0.1, K=7, embedding_size=InputsConfig.EMBEDDING_DIM)
    _ = process_embeddings('arizona', weight=0.1, K=7, embedding_size=InputsConfig.EMBEDDING_DIM)
    _ = process_embeddings('georgia', weight=0.1, K=7, embedding_size=InputsConfig.EMBEDDING_DIM)
    _ = process_embeddings('florida', weight=0.1, K=7, embedding_size=InputsConfig.EMBEDDING_DIM)
    _ = process_embeddings('california', weight=0.1, K=7, embedding_size=InputsConfig.EMBEDDING_DIM)
    _ = process_embeddings('texas', weight=0.1, K=7, embedding_size=InputsConfig.EMBEDDING_DIM)
