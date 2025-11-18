from src.configs.paths import IO_CHECKINS
from src.etl.create_inputs_hgi import process_state
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa estado e gera embeddings")
    parser.add_argument(
        "state_name",
        type=str,
        help="Nome do estado (ex: montana, florida, california)"
    )
    args = parser.parse_args()

    state_name = args.state_name

    process_state(state_name)