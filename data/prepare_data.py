from datasets import load_dataset
# from utils.config import DATA_PATH, MAX_TRAIN_SAMPLES, MAX_EVAL_SAMPLES
import os

MAX_TRAIN_SAMPLES = 5000
MAX_EVAL_SAMPLES = 500

def prepare_data():
    dataset = load_dataset("microsoft/ms_marco", "v2.1", cache_dir="data_files")
    train_dataset = dataset['train'].select(range(MAX_TRAIN_SAMPLES))
    validation_dataset = dataset['validation'].select(range(MAX_EVAL_SAMPLES))
    train_dataset.save_to_disk(os.path.join("data_files", 'train_dataset'))
    validation_dataset.save_to_disk(os.path.join("data_files", 'validation_dataset'))

if __name__ == "__main__":
    prepare_data()
