import numpy as np
# import matplotlib as mpl
# mpl.use("TKAgg")
import matplotlib.pyplot as plt
import sys
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import Buffer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists('results'):
    os.makedirs('results')


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # Task setup block starts
    # Do not change
    torch.manual_seed(1000)
    dataset = datasets.MNIST(
        root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    # Task setup block end

    # Learner setup block
    seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])
    torch.manual_seed(seed)  # do not change. This is for learners randomization
    ####### Start
    # net = Network(28 * 28, 10, 64)
    buffer_capacity = 100
    N = 10
    update_freq = 50  # this should be >= buffer_capacity
    epochs = 10
    buffer = Buffer(buffer_capacity, [1, 28, 28])
    net = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0003)
    ####### End

    # Experiment block starts
    errors = []
    checkpoint = 1000
    correct_pred = 0
    for idx, (image, label) in enumerate(loader):
        # Observe
        label = label.to(device=device)
        image = image.to(device=device)
        # Make a prediction of label
        ####### Start
        # Replace the following statement with your own code for
        # making label prediction
        buffer.add(image.detach().numpy(), label.detach().numpy())
        with torch.no_grad():
            pred = net(image)
            pred_label = torch.argmax(pred, axis=1) if len(pred.shape) > 1 else torch.argmax(pred)
        ####### End

        # Evaluation

        correct_pred += (pred_label == label).sum()

        # Learn
        ####### Start
        # Here goes your learning update
        if (idx + 1) % update_freq == 0:
            for _ in range(epochs):
                image_mini_batches, label_mini_batches = buffer.sample(N)
                for image_batch, label_batch in zip(image_mini_batches, label_mini_batches):
                    image_batch = torch.FloatTensor(image_batch).to(device)
                    label_batch = torch.LongTensor(label_batch).to(device)
                    pred_batch = net(image_batch)
                    optimizer.zero_grad()
                    loss = criterion(pred_batch, label_batch)
                    loss.backward()
                    optimizer.step()
        ####### End

        # Log
        if (idx + 1) % checkpoint == 0:
            error = float(correct_pred) / float(checkpoint) * 100
            print(error)
            errors.append(error)
            correct_pred = 0
            plt.clf()
            plt.plot(range(checkpoint, (idx + 1) + checkpoint, checkpoint), errors)
            plt.ylim([0, 100])
            plt.pause(0.001)
    name = sys.argv[0].split('.')[-2].split('_')[-1]
    data = np.zeros((2, len(errors)))
    data[0] = range(checkpoint, 60000 + 1, checkpoint)
    data[1] = errors
    np.savetxt(os.path.join('results', name + str(seed) + ".txt"), data)
    plt.show()


if __name__ == "__main__":
    main()
