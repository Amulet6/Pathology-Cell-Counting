import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, 'checkpoints')
LOG_ROOT = os.path.join(PROJECT_ROOT, 'logs')

DATASETS = {
    'bcdata': {'path': os.path.join(DATA_ROOT, 'BCData')},
    'conic': {'path': os.path.join(DATA_ROOT, 'CoNIC')},
    'monuseg': {'path': os.path.join(DATA_ROOT, 'MoNuSeg')},
}

TRAIN_CONFIG = {
    'batch_size': 16,
    'epochs': 100,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'seed': 42,
}
