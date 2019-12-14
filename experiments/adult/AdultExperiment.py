import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from Dataloader import AdultDataset
from models import SlowDMonotonicNN
from tensorboardX import SummaryWriter


writer = SummaryWriter()
train_ds = AdultDataset("../../data/adult/adult.data")
test_ds = AdultDataset("../../data/adult/adult.test", test=True)

train_dl = DataLoader(train_ds, 100, shuffle=True, num_workers=4)
test_dl = DataLoader(test_ds, 100, shuffle=True, num_workers=4)

x, y = train_ds[1]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#net = nn.Sequential(nn.Linear(len(x), 200), nn.ReLU(), nn.Linear(200, 200), nn.ReLU(), nn.Linear(200, 1))#
net = SlowDMonotonicNN(4, len(x) - 4, [150, 150, 150, 150], 1, device)
net.load_state_dict(torch.load("model.ckpt"))
optim = Adam(net.parameters(), lr=.001)
loss_f = nn.BCELoss()
sigmoid = nn.Sigmoid()

for epoch in range(1000):
    avg_loss = 0.
    i = 0
    avg_accuracy = 0.
    for x, y in train_dl:
        x,y = x.float().to(device), y.float().to(device)
        y_est = sigmoid(net(x[:, :4], x[:, 4:])).squeeze(1)
        loss = loss_f(y_est, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        avg_loss += loss.item()
        avg_accuracy += torch.abs((y_est.detach() > .5).float() == y).float().mean()
        i += 1
        if i % 100 == 0:
            print(i)
    writer.add_scalars("Adult/BCE", {"train": avg_loss / i}, epoch)
    writer.add_scalars("Adult/Accuracy", {"train": avg_accuracy / i}, epoch)
    print("train", epoch, avg_loss / i, avg_accuracy / i)
    avg_loss = 0.
    i = 0
    avg_accuracy = 0.
    for x, y in test_dl:
        with torch.no_grad():
            x, y = x.float().to(device), y.float().to(device)
            y_est = sigmoid(net(x[:, :4], x[:, 4:])).squeeze(1)
            loss = loss_f(y_est, y)
            avg_loss += loss.item()
            avg_accuracy += torch.abs((y_est > .5).float() == y).float().mean()
            i += 1
            if i % 100 == 0:
                print(i)
    writer.add_scalars("Adult/BCE", {"test": avg_loss / i}, epoch)
    writer.add_scalars("Adult/Accuracy", {"test": avg_accuracy / i}, epoch)
    print("test", epoch, avg_loss / i, avg_accuracy / i)
    torch.save(net.cpu().state_dict(), "model.ckpt")
    print(epoch, avg_loss/i, avg_accuracy/i)

