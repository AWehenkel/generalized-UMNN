import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from Dataloader import AdultDataset
from models import SlowDMonotonicNN

train_ds = AdultDataset("../../data/adult/adult.data")
test_ds = AdultDataset("../../data/adult/adult.test")

train_dl = DataLoader(train_ds, 10, shuffle=True, num_workers=4)
test_dl = DataLoader(test_ds, 10, shuffle=True, num_workers=4)

x, y = train_ds[1]

net = nn.Sequential(nn.Linear(len(x), 200), nn.ReLU(), nn.Linear(200, 200), nn.ReLU(), nn.Linear(200, 1))#SlowDMonotonicNN(4, len(x) - 4, [50, 50, 50], 1, "cpu")
optim = Adam(net.parameters(), lr=.001)
loss_f = nn.BCELoss()
sigmoid = nn.Sigmoid()

for epoch in range(1000):
    avg_loss = 0.
    i = 0
    avg_accuracy = 0.
    for x, y in train_dl:
        x,y = x.float(), y.float()
        y_est = sigmoid(net(x))#sigmoid(net(x[:, :4], x[:, 4:]))
        loss = loss_f(y_est, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        avg_loss += loss.item()
        avg_accuracy += torch.abs((y_est.detach() > .5).float() == y).float().mean()
        print((y_est.detach() >= .5).float() - y)
        i += 1
        print(i)
    print(avg_loss/i, avg_accuracy/i)

