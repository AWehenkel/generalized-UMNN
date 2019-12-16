import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from experiments.adult.Dataloader import AdultDataset
from models import SlowDMonotonicNN
from tensorboardX import SummaryWriter

class EmbeddingNet(nn.Module):
    def __init__(self, in_embedding, in_main, out_embedding, device):
        super(EmbeddingNet, self).__init__()
        self.embedding_net = nn.Sequential(nn.Linear(in_embedding, 200), nn.ReLU(),
                                      nn.Linear(200, 200), nn.ReLU(),
                                      nn.Linear(200, out_embedding), nn.ReLU()).to(device)
        self.umnn = SlowDMonotonicNN(in_main, out_embedding, [100, 100, 100], 1, 300, device)

    def set_steps(self, nb_steps):
        self.umnn.set_steps(nb_steps)

    def forward(self, x):
        h = self.embedding_net(x[:, 4:])
        return torch.sigmoid(self.umnn(x[:, :4], h))


class SimpleMLP(nn.Module):
    def __init__(self, in_embedding, in_main, out_embedding, device):
        super(SimpleMLP, self).__init__()
        self.embedding_net = nn.Sequential(nn.Linear(in_embedding, 100), nn.ReLU(),
                                      nn.Linear(100, 100), nn.ReLU(),
                                      nn.Linear(100, out_embedding), nn.ReLU()).to(device)
        self.mlp = nn.Sequential(nn.Linear(in_main + out_embedding, 100), nn.ReLU(),
                                  nn.Linear(100, 100), nn.ReLU(),
                                  nn.Linear(100, 1), nn.Sigmoid()).to(device)

    def set_steps(self, nb_steps):
        return

    def forward(self, x):
        h = self.embedding_net(x[:, 4:])
        return self.mlp(torch.cat((x[:, :4], h), 1))


def run_adult_experiment():
    writer = SummaryWriter()
    train_ds = AdultDataset("data/adult/adult.data", normalization=True)
    test_ds = AdultDataset("data/adult/adult.test", test=True, normalization=False)

    mu, std = train_ds.mu, train_ds.std
    test_ds.normalize(mu, std)

    train_dl = DataLoader(train_ds, 100, shuffle=False, num_workers=1)
    test_dl = DataLoader(test_ds, 100, shuffle=False, num_workers=1)

    x, y = train_ds[1]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    net = EmbeddingNet(len(x) - 4, 4, 30, device)
    #net = SimpleMLP(len(x) - 4, 4, 30, device)


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

    for epoch in range(1000):
        avg_loss = 0.
        i = 0
        avg_accuracy = 0.
        for x, y in train_dl:
            x,y = x.float().to(device), y.unsqueeze(1).float().to(device)
            y_est = net(x)
            loss = loss_f(y_est, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            avg_loss += loss.item()
            avg_accuracy += torch.abs((y_est.detach() > .5).float() == y).float().mean()
            net.set_steps(int(torch.randint(30, 60, [1])))
            #net.set_steps(100)

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
                y_est = net(x)
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

