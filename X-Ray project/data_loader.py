import os
from preprocessing import get_data_generators

def load_data(base_dir):
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
    
    train_gen, val_gen, test_gen = get_data_generators(train_dir, val_dir, test_dir)
    return train_gen, val_gen, test_gen