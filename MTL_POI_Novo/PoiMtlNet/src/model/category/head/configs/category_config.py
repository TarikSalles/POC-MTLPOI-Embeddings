from configs.model import InputsConfig


class CfgCategoryHyperparams:
    LR = 0.0001
    MAX_LR = 0.01
    WEIGHT_DECAY = 5e-2
    MAX_GRAD_NORM = 1.0


class CfgCategoryModel:
    INPUT_DIM = InputsConfig.EMBEDDING_DIM
    HIDDEN_DIMS = (256, 128, 64, 32)
    NUM_CLASSES = 7
    DROPOUT = 0.2


class CfgCategoryTraining:
    BATCH_SIZE = 2**10  # 512
    EPOCHS = 100
    K_FOLDS = 5
    SEED = 42