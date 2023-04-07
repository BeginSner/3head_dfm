
import torch
from torch.utils.data import TensorDataset, DataLoader
from data import get_criteo_dataset
from models import get_model
# Disable gradient computation messages
torch.set_grad_enabled(False)

# assign the GPU
physical_devices = torch.cuda.device_count()

# now only use single GPU
torch.cuda.set_device(0)
torch.cuda.empty_cache()

def run(params):
    dataset = get_criteo_dataset(params)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_data = TensorDataset(
        torch.tensor(train_dataset["x"]), torch.tensor(train_dataset["labels"]))
    train_data = DataLoader(
        train_data,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    test_data = TensorDataset(
        torch.tensor(test_dataset["x"]), torch.tensor(test_dataset["labels"])
    )

    model = get_model(params["model"], params)
    optimizer = get
