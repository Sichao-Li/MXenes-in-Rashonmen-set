import torch
from torch.nn import Linear, MSELoss
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import Dataset, TensorDataset
import numpy as np
from src.utilities import load_data, preprocess
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class Net_mx_descriptor(torch.nn.Module):
    def __init__(self):
        super(Net_mx_descriptor, self).__init__()
        self.fc1 = Linear(4, 64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 64)
        self.fc4 = Linear(64, 5)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.fc4(x)
        # x, idx = x.sort()
        return x

if __name__ == "__main__":
    train_dataloader, test_dataloader, val_dataloader = preprocess(elemental_descriptor=False)
    model = Net_mx_descriptor()
    critereon = MSELoss()
    # define the optimizer
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    # define the number of epochs and the data set size
    nb_epochs = 500
    min_valid_loss = np.inf
    train_loss = []
    val_loss = []
    for epoch in range(nb_epochs):
        model.train()
        epoch_loss = 0
        for batch_index, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = critereon(y_pred, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            # print("Epoch: {} Loss: {}".format(epoch, epoch_loss))
        train_loss.append(np.array(epoch_loss))
        valid_loss = 0.0
        model.eval()
        for batch_index, (inputs, labels) in enumerate(val_dataloader):
            y_pred = model(inputs)
            loss = critereon(y_pred, labels)
            valid_loss += loss.item()
        val_loss.append(np.array(valid_loss))
        if min_valid_loss > valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), '../results/model/best_model_ANN_mx_descriptor.pth')
        print("Epoch: {} Loss: {} Val_Loss: {}".format(epoch, epoch_loss, valid_loss))
    with open('../results/model/train_loss_mx_descriptor.npy', 'wb') as f:
        np.save(f, train_loss)
    with open('../results/model/test_loss_mx_descriptor.npy', 'wb') as f:
        np.save(f, val_loss)