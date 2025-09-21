from datasets import load_dataset, load_from_disk

data_dir = r"G:\hugging-face-datasets"

# download
# dataset = load_dataset(path="lansinuote/ChnSentiCorp", cache_dir=data_dir)
# dataset.save_to_disk(r"G:\hugging-face-datasets\lansinuote2")

# load
dataset = load_from_disk(r"G:\hugging-face-datasets\lansinuote2")

for data in dataset['train']:
    print(data)
    # break