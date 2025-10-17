from configs.model import InputsConfig


class CfgNextHyperparams:
    LR = 0.0001
    MAX_LR = 0.01
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1.0


class CfgNextModel:
    INPUT_DIM = InputsConfig.EMBEDDING_DIM
    NUM_HEADS = 8
    NUM_LAYERS = 4
    MAX_SEQ_LENGTH = 9
    NUM_CLASSES = 7
    DROPOUT = 0.1


class CfgNextTraining:
    BATCH_SIZE = 2**10  # 512
    EPOCHS = 100
    K_FOLDS = 5
    SEED = 42