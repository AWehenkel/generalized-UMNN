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
    test_ds = AdultDataset("data/adult/adult.test", test=True, normalization=False)

    mu, std = train_ds.mu, train_ds.std
    test_ds.normalize(mu, std)

    train_dl = DataLoader(train_ds, 100, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, 100, shuffle=True, num_workers=4)

    x, y = train_ds[1]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedding_net = nn.Sequential(nn.Linear(len(x) - 4, 200), nn.ReLU(),
                                  nn.Linear(200, 200), nn.ReLU(),
                                  nn.Linear(200, 100), nn.ReLU()).to(device)
    net = SlowDMonotonicNN(4, 100, [100, 100, 100], 1, 300, device)
    if False:
        net.load_state_dict(torch.load("model.ckpt"))
        x = torch.randn(500, 4)
        h = torch.zeros(500, 100)
        with torch.no_grad():
            import matplotlib.pyplot as plt
            plt.subplot(221)
            y = net(torch.cat((x[:, [0]], torch.zeros(500, 3)), 1), h)
            plt.scatter(x[:, 0], y)
            plt.subplot(222)
            y = net(torch.cat((torch.zeros(500, 1), x[:, [1]], torch.zeros(500, 2)), 1), h)
            plt.scatter(x[:, 1], y)
            plt.subplot(223)
            y = net(torch.cat((torch.zeros(500, 2), x[:, [2]], torch.zeros(500, 1)), 1), h)
            plt.scatter(x[:, 2], y)
            plt.subplot(224)
            y = net(torch.cat((torch.zeros(500, 3), x[:, [3]]), 1), h)
            plt.scatter(x[:, 3], y)
        plt.show()
        exit()
    optim = Adam(net.parameters(), lr=.001, weight_decay=1e-5)
    loss_f = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    for epoch in range(1000):
        avg_loss = 0.
        i = 0
        avg_accuracy = 0.
        for x, y in train_dl:
            x,y = x.float().to(device), y.unsqueeze(1).float().to(device)
            h = embedding_net(x[:, 4:])
            y_est = sigmoid(net(x[:, :4], h))
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
                x, y = x.float().to(device), y.unsqueeze(1).float().to(device)
                h = embedding_net(x[:, 4:])
                y_est = sigmoid(net(x[:, :4], h))
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

