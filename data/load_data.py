from datasets import load_from_disk
# from utils.config import DATA_PATH

DATA_PATH = "data_files"

def load_train_data():
    return load_from_disk(f"{DATA_PATH}/train_dataset")

def load_eval_data():
    return load_from_disk(f"{DATA_PATH}/validation_dataset")
