import torch
from torch.nn import Linear, MSELoss
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import Dataset, TensorDataset
import sys
sys.path.append(r"../src")
from Utilities import *
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

def preprocess():
    X_df = load_data(filename='../data/data_withLattice.csv')
    X_features = X_df.iloc[:, 1:-5]
    feature_names = X_features.columns.values

    y_multilabel = X_df.iloc[:, -5:]
    label_names = y_multilabel.columns.values

    X = X_features.to_numpy()
    y = y_multilabel.to_numpy()
    scaler = preprocessing.MinMaxScaler()
    y = scaler.fit_transform(y)

    data_index = []
    for index, row in X_features.iterrows():
        indices = [i for i, x in enumerate(row) if x == 1]
        indices[1] = indices[1] - 9
        indices[2] = indices[2] - 11
        indices[3] = indices[3] - 16
        data_index.append(indices)
    X_features_classes = np.array(data_index)

    X_train, X_test, y_train, y_test = train_test_split(X_features_classes, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.Tensor(y_test)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    return train_dataloader, test_dataloader

if __name__ == "__main__":
    train_dataloader, test_dataloader = preprocess()
    model = Net()
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
        for batch_index, (inputs, labels) in enumerate(test_dataloader):
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