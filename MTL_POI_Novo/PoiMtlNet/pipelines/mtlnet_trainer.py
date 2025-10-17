import os
import pickle

import joblib
import torch

from configs.model import MTLModelConfig
from configs.paths import OUTPUT_ROOT, RESULTS_ROOT
from src.data.create_fold import create_folds

import logging

from src.model.mtlnet.engine.mtl_train import train_with_cross_validation
from utils.ml_history.metrics import MLHistory
from utils.ml_history.utils.dataset import DatasetHistory

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    state = "montana"  # Replace with the desired state
    output_dir = f'{OUTPUT_ROOT}/{state}/pre-processing'
    # output_dir = f'/Users/vitor/Desktop/mestrado/ingred/data/ori/pre-processing/chicago'

    # Define parameters
    next_data_path = f'{output_dir}/next-input.csv'  # Replace with actual path
    category_data_path = f'{output_dir}/category-input.csv'  # Replace with actual path

    logging.info(f'Creating folds')

    # Process data and create dataloaders
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
                description="Data related to next POI prediction. Data with 107 features",
            ),
            DatasetHistory(
                raw_data=category_data_path,
                folds_signature=folds_path,
                description="Data related to category prediction. Data with 107 features",
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

    history.storage.save(path=RESULTS_ROOT + f'/{state}')
