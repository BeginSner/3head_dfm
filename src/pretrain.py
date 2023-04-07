import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from metrics import cal_llloss_with_logits, cal_auc, cal_prauc
from data import get_criteo_dataset
from models import get_model
from utils import get_optimizer
from loss import get_loss_fn

# Disable gradient computation messages
torch.set_grad_enabled(False)

# assign the GPU
physical_devices = torch.cuda.device_count()

# now only use single GPU
torch.cuda.set_device(0)
torch.cuda.empty_cache()


def test(model, test_data, params):
    all_logits = []
    all_probs = []
    all_labels = []
    for step, (batch_x, batch_y) in enumerate(tqdm(test_data), 1):
        logits = model(batch_x.to(params['device']), training=False)["logits"]
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(batch_y.numpy())
        all_probs.append(torch.sigmoid(logits))
    all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1, 1))
    all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1, 1))
    all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1, 1))
    llloss = cal_llloss_with_logits(all_labels, all_logits)
    auc = cal_auc(all_labels, all_probs)
    prauc = cal_prauc(all_labels, all_probs)
    batch_size = all_logits.shape[0]
    return auc


def optim_step(model, x, targets, optimizer, loss_fn, params):
    optimizer.zero_grad()  # Reset gradients to zero
    outputs = model(x)
    reg_loss = sum(model.losses())
    loss_dict = loss_fn(targets, outputs, params)
    loss = loss_dict["loss"] + reg_loss
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights


def train(model, optimizer, train_data, params):
    for step, batch in enumerate(tqdm(train_data), 1):
        batch_x = batch[0]
        batch_y = batch[1]
        targets = {"label": batch_y}
        optim_step(model, batch_x, targets, optimizer,
                   get_loss_fn(params["loss"]), params)


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
    optimizer = get_optimizer(params["optimizer"], params)
    best_acc = 0
    for ep in range(params["epoch"]):
        train(model, optimizer, train_data, params)
        model.save_weights(params["model_ckpt_path"], save_format="tf")
