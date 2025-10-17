import argparse
import os
import random
import time
from typing import Optional

import joblib

from configs.paths import OUTPUT_ROOT, RESULTS_ROOT

from model.next.head.configs.next_config import CfgNextTraining, CfgNextHyperparams
from model.next.head.data.fold import load_data, create_folds
from model.next.head.engine.cross_validation import run_cv
from utils.ml_history.metrics import MLHistory
from utils.ml_history.utils.dataset import DatasetHistory

if __name__ == '__main__':
    # As args get the epochs, batch size and learning rate
    parser = argparse.ArgumentParser(description='Train the next head of the model')
    parser.add_argument('--ep', type=int, default=CfgNextTraining.EPOCHS, help='Number of epochs to train')
    parser.add_argument('--bs', type=int, default=CfgNextTraining.BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=CfgNextHyperparams.LR, help='Learning rate for the optimizer')

    parser.add_argument('--state', type=str, default='florida_dgi_new', help='State to train the model on')
    parser.add_argument('--save-folds', type=bool, default=False, help='Whether to save the folds in a pickle file')
    parser.add_argument('--folds-chkpt', type=Optional[str], default=None, help='Path to the checkpoint file for the folds')

    args = parser.parse_args()
    state = args.state
    input_dir = f'{OUTPUT_ROOT}/{state}'
    data_input = f'{input_dir}/pre-processing/next-input.csv'
    output_dir = f'{RESULTS_ROOT}/{state}'

    # Creating folds
    folds = None
    if args.folds_chkpt is not None:
        folds_chkpt = f'{output_dir}/{args.folds_chkpt}'
        with open(folds_chkpt, 'rb') as f:
            folds = joblib.load(f)
    else:
        X, y = load_data(data_input)
        folds = create_folds(
            X,
            y,
            n_splits=CfgNextTraining.K_FOLDS,
            batch_size=args.bs,
            seed=random.randint(1, 10000),
        )
        if args.save_folds:
            folds_pth = os.path.join(input_dir, 'folds')
            os.makedirs(folds_pth, exist_ok=True)
            fold_file = os.path.join(folds_pth,time.strftime("%Y%m%d_%H%M") + "_folds.pkl")
            with open(fold_file, 'wb') as f:
                joblib.dump(folds, f)

    # Creating history
    history: MLHistory = MLHistory(
        model_name="Next",
        model_type="Single-Task",
        tasks='next',
        num_folds=CfgNextTraining.K_FOLDS,
        datasets={
            DatasetHistory(
                raw_data=data_input,
                folds_signature=args.folds_chkpt or None,
                description="POI next Classification",
            )
        }
    )

    # Running cross-validation
    with history.context() as history:
        run_cv(history, folds)

    history.display.end_training()
    history.storage.save(path=output_dir)









