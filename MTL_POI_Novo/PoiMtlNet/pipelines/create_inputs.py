import argparse
import os
from configs.paths import IO_CHECKINS
from src.data.create_embb import create_embeddings
from src.data.create_input import process_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa estado e gera embeddings")
    parser.add_argument(
        "state_name",
        type=str,
        help="Nome do estado (ex: montana, florida, california)"
    )
    args = parser.parse_args()

    state_name = args.state_name.lower().strip()

    process_state(state_name)
