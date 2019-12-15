import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from experiments.adult.Dataloader import AdultDataset
from models import SlowDMonotonicNN
from tensorboardX import SummaryWriter


def run_adult_experiment():
    writer = SummaryWriter()
    train_ds = AdultDataset("data/adult/adult.data")
    test_ds = AdultDataset("data/adult/adult.test", test=True)

    train_dl = DataLoader(train_ds, 100, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, 100, shuffle=True, num_workers=4)

    x, y = train_ds[1]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedding_net = nn.Sequential(nn.Linear(len(x) - 4, 200), nn.ReLU(), nn.Linear(200, 200), nn.ReLU(), nn.Linear(200, 20)).to(device)
    net = SlowDMonotonicNN(4, 20, [100, 100, 100], 1, 100, device)
    #net.load_state_dict(torch.load("model.ckpt"))
    optim = Adam(net.parameters(), lr=.001, weight_decay=1e-5)
    loss_f = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    for epoch in range(1000):
        avg_loss = 0.
        i = 0
        avg_accuracy = 0.
        for x, y in train_dl:
            x,y = x.float().to(device), y.float().to(device)
            h = embedding_net(x[:, 4:])
            y_est = sigmoid(net(x[:, :4], h)).squeeze(1)
            loss = loss_f(y_est, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            avg_loss += loss.item()
            avg_accuracy += torch.abs((y_est.detach() > .5).float() == y).float().mean()
            net.set_steps(int(torch.randint(30, 60, [1])))
            i += 1
            if i % 100 == 0:
                print(i)
        writer.add_scalars("Adult/BCE", {"train": avg_loss / i}, epoch)
        writer.add_scalars("Adult/Accuracy", {"train": avg_accuracy / i}, epoch)
        print("train", epoch, avg_loss / i, avg_accuracy / i)
        avg_loss = 0.
        i = 0
        avg_accuracy = 0.
        net.set_steps(100)
        for x, y in test_dl:
            with torch.no_grad():
                x, y = x.float().to(device), y.float().to(device)
                h = embedding_net(x[:, 4:])
                y_est = sigmoid(net(x[:, :4], h)).squeeze(1)
                loss = loss_f(y_est, y)
                avg_loss += loss.item()
                avg_accuracy += torch.abs((y_est > .5).float() == y).float().mean()
                i += 1
                if i % 100 == 0:
                    print(i)
        writer.add_scalars("Adult/BCE", {"test": avg_loss / i}, epoch)
        writer.add_scalars("Adult/Accuracy", {"test": avg_accuracy / i}, epoch)
        print("test", epoch, avg_loss / i, avg_accuracy / i)
        torch.save(net.state_dict(), "model.ckpt")

