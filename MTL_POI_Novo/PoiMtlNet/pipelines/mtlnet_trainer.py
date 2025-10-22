import os
import pickle
import joblib
import torch
import argparse
import logging

from configs.model import MTLModelConfig
from configs.paths import OUTPUT_ROOT, RESULTS_ROOT
from src.data.create_fold import create_folds
from src.model.mtlnet.engine.mtl_train import train_with_cross_validation
from utils.ml_history.metrics import MLHistory
from utils.ml_history.utils.dataset import DatasetHistory

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Treina MTLNet para um estado específico")
    parser.add_argument(
        "state",
        type=str,
        help="Nome do estado (ex: montana, california, florida)"
    )
    args = parser.parse_args()

    state = args.state.lower().strip()
    logging.info(f"Iniciando treinamento para o estado: {state}")

    output_dir = f'{OUTPUT_ROOT}/{state}/pre-processing'

    next_data_path = f'{output_dir}/next-input.csv'
    category_data_path = f'{output_dir}/category-input.csv'

    if not os.path.exists(next_data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {next_data_path}")
    if not os.path.exists(category_data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {category_data_path}")

    logging.info(f'Criando folds para {state}')

    fold_results, folds_path = create_folds(
        next_data_path,
        category_data_path,
        k_splits=MTLModelConfig.K_FOLDS,
        save_folder=None,
    )

    history = MLHistory(
        model_name='MTLNet',
        tasks={'next', 'category'},
        num_folds=MTLModelConfig.K_FOLDS,
        datasets={
            DatasetHistory(
                raw_data=next_data_path,
                folds_signature=folds_path,
                description=f"Next POI data for {state}."
            ),
            DatasetHistory(
                raw_data=category_data_path,
                folds_signature=folds_path,
                description=f"Category prediction data for {state}."
            )
        }
    )

    with history.context() as history:
        results = train_with_cross_validation(
            dataloaders=fold_results,
            history=history,
            num_classes=MTLModelConfig.NUM_CLASSES,
            num_epochs=MTLModelConfig.EPOCHS,
            learning_rate=MTLModelConfig.LEARNING_RATE
        )

    save_path = os.path.join(RESULTS_ROOT, state)
    os.makedirs(save_path, exist_ok=True)
    history.storage.save(path=save_path)

    logging.info(f"Treinamento concluído com sucesso para {state}. Resultados salvos em {save_path}")
