# Please be in the same directory as the data files
# cd [directory]

# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

## *To see the DataFrame
# df = pd.read_csv("./covid_train.csv")
# print(df.describe)
# df_test = pd.read_csv("./covid_test.csv")
# print(df_test.describe)

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter

# Plotting curves
import matplotlib.pyplot as plt

#####
train_loss = []
test_loss = []
use_tensorboard = False
#####


def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    """Split provided training data into training set and validation set"""
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(
        data_set,
        [train_set_size, valid_set_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval()  # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


class COVID19Dataset(Dataset):
    """
    x: Features.
    y: Targets, if none, do prediction.
    """

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


# 一个简单的多层感知机（MLP），包含三个全连接层和两个 ReLU 激活函数。--sample code
# This model can make the scores of strong:0.90
# Now I select different features and i match the score: 0.85
# but the boss-line: 0.81456 strone:0.92619
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        # * modify 1st: ReLU() -> LeakyReLU(a=0.01) : Leaky ReLU play better than ReLU in small dataset.
        # * modify 2nd: add BatchNorm1d() :  BatchNorm1d() will make the data frame more suitable to process when it comes to mult-column.
        # * modify 3rd: Dropout(a=0.2~0.5) : to smash some unused fuction.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 1),
            # nn.Dropout(0.1) # ? why add this will make loss go to inf XDDDD
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x


def select_feat(train_data, valid_data, test_data, select_all=True):
    """Selects useful features to perform regression"""
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = (
        train_data[:, :-1],
        valid_data[:, :-1],
        test_data,
    )
    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = list(
            # range(35, raw_x_train.shape[1])
            # [
            #     34,
            #     36,
            #     51,
            #     52,
            #     54,
            #     70,
            #     72,
            #     69,
            # ]  # * This is what the feature sayng in this ppt.
            # [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
            # [
            #     35,
            #     36,
            #     37,
            #     47,
            #     48,
            #     52,
            #     53,
            #     54,
            #     55,
            #     65,
            #     66,
            #     70,
            #     71,
            #     72,
            #     73,
            # ]
            [
                34,
                35,
                36,
                46,
                47,
                51,
                52,
                53,
                54,
                64,
                65,
                69,
                70,
                71,
                72,
                82,
                83,
                87,
            ]
        )  # TODO: Select suitable feature columns.

    return (
        raw_x_train[:, feat_idx],
        raw_x_valid[:, feat_idx],
        raw_x_test[:, feat_idx],
        y_train,
        y_valid,
    )


def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(
        reduction="mean"
    )  # Define your loss function, do not modify this.
    # Define your optimization algorithm.
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=0.96,  # 一阶动量
        nesterov=True,  # Nesterov动量
        weight_decay=6e-4,  # L2正则化
        dampening=0.0,  # dampening 同时与上面的动量、学习率合计运算，启用nesterov动量要求momentum>0,dampening=0
    )
    if use_tensorboard:
        writer = SummaryWriter()  # Writer of tensoboard.

    if not os.path.isdir("./models"):
        os.mkdir("./models")  # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config["n_epochs"], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f"Epoch [{epoch+1}/{n_epochs}]")
            train_pbar.set_postfix({"loss": loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        if use_tensorboard:
            writer.add_scalar("Loss/train", mean_train_loss, step)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(
            f"Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}"
        )
        if use_tensorboard:
            writer.add_scalar("Loss/valid", mean_valid_loss, step)
        ####################
        # Visual loss curve. I can see this by tensorboard
        train_loss.append(mean_train_loss)
        test_loss.append(mean_valid_loss)
        ####################

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config["save_path"])  # Save your best model
            print("Saving model with loss {:.3f}...".format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config["early_stop"]:
            print("\nModel is not improving, so we halt the training session.")
            return


device = "cuda" if torch.cuda.is_available() else "cpu"
config = {
    "seed": 3407,  # Your seed number, you can pick your lucky number. :)
    "select_all": False,  # Whether to use all features. True -> False
    "valid_ratio": 0.2,  # validation_size = train_size * valid_ratio
    "n_epochs": 2000,  # Number of epochs.
    "batch_size": 256,
    "learning_rate": 7e-6,
    "early_stop": 600,  # If model has not improved for this many consecutive epochs, stop training.
    "save_path": "./models/model.ckpt",  # Your model will be saved here.
}

# Set seed for reproducibility
same_seed(config["seed"])


# train_data size: 3009 x 89 (35 states + 18 features x 3 days)
# 35 + 18 x 3 = 35 + 54 = 89
# test_data size: 997 x 88 (without last day's positive rate)
train_data, test_data = (
    pd.read_csv("./covid_train.csv").values,
    pd.read_csv("./covid_test.csv").values,
)
train_data, valid_data = train_valid_split(
    train_data, config["valid_ratio"], config["seed"]
)

# Print out the data size.
print(
    f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}"""
)

# Select features
x_train, x_valid, x_test, y_train, y_valid = select_feat(
    train_data, valid_data, test_data, config["select_all"]
)

# Print out the number of features.
print(f"number of features: {x_train.shape[1]}")

train_dataset, valid_dataset, test_dataset = (
    COVID19Dataset(x_train, y_train),
    COVID19Dataset(x_valid, y_valid),
    COVID19Dataset(x_test),
)

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(
    train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True
)

model = My_Model(input_dim=x_train.shape[1]).to(
    device
)  # put your model and data on the same computation device.

# train
trainer(train_loader, valid_loader, model, config, device)
plt.show()
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, label="train loss", linestyle=":")
plt.plot(epochs, test_loss, label="valid loss", linestyle="-.")
plt.yscale("log")
# plt.ylim(bottom=0.75, top=2.0)
plt.ylabel("loss(log)")
plt.xlabel("epochs")
plt.title("Loss")
plt.legend()
plt.savefig("./loss.jpg")


# test
def save_pred(preds, file):
    """Save predictions to specified file"""
    with open(file, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "tested_positive"])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


# pred
model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config["save_path"]))

preds = predict(test_loader, model, device)
save_pred(preds, "./pred.csv")
