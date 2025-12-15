# setup
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


# Loading Data
def get_dataloaders(batch_size, p=0.7):
    # Load dataset
    X_train = np.load(os.path.join("data", "X_train.npy"))
    y_train = np.load(os.path.join("data", "y_train.npy"))

    X_val = np.load(os.path.join("data", "X_val.npy"))
    y_val = np.load(os.path.join("data", "y_val.npy"))

    X_test = np.load(os.path.join("data", "X_test.npy"))
    y_test = np.load(os.path.join("data", "y_test.npy"))

    # normalization of the data
    mean = X_train.mean(0)
    std = X_train.std(0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    X_val = (X_val - mean) / std

    train_loader = DataLoader(
        dataset=TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        dataset=TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, validation_loader, test_loader, mean, std


train_loader, validation_loader, test_loader, mean, std = get_dataloaders(batch_size=64)

classes = (
    "asymptomatic",
    "affected",
)


def show_gait(loader, class_labels, title=""):
    """Show n images per class"""
    class_count = len(class_labels)

    x = loader.dataset.tensors[0]
    y = loader.dataset.tensors[1]

    x_0 = x[y == 0, :]  # asymptomatic
    x_1 = x[y == 1, :]  # affected

    t = list(range(x.shape[1]))
    mu_0 = x_0.mean(0)
    std_0 = x_0.std(0)

    mu_1 = x_1.mean(0)
    std_1 = x_1.std(0)

    plt.plot(t, mu_1, lw=2, label="affected", color="#5387DD")
    plt.fill_between(t, mu_1 + std_1, mu_1 - std_1, facecolor="#5387DD", alpha=0.3)
    plt.plot(t, mu_0, lw=2, label="asymptomatic", color="#C565C7")
    plt.fill_between(t, mu_0 + std_0, mu_0 - std_0, facecolor="#C565C7", alpha=0.3)
    plt.legend()
    plt.title(title)

    plt.show()


show_gait(train_loader, classes, title="Training Data")
show_gait(test_loader, classes, title="Test Data")
show_gait(validation_loader, classes, title="Validation Data")


import torch.nn.functional as F


# Defining an RNN model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        # Define your new architecture here

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Define the forward pass
        x = x.view(x.shape[0], x.shape[1], 1)
        out = self.rnn(x)[0]
        out = out[:, -1, :]
        out = self.output_layers(out)
        return out


def evaluate(model, data_loader, loss_fn, device):
    losses = []
    acc = []

    model.eval()
    for i, data in enumerate(data_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # place data in the correct device
        inputs = inputs.to(device)
        labels = labels.to(device).view(-1, 1)

        with torch.no_grad():
            # forward
            outputs = model(inputs)

            # loss
            loss = loss_fn(outputs, labels)

            # accuracy
            accuracy = (outputs.round) == labels

            # keep loss for plotting
            losses.append(loss.item())
            # keep accuracy for plotting
            acc.append(loss.item())

    return np.mean(losses), np.mean(acc)


np.random.seed(13)

# define hyperparameters
epochs = 10
batch_size = 10
learning_rate = 1e-2
eval_every = 1  # number of epochs between evaluations
evaluate_testset_during_training = (
    True  # whether to evaluate the testset during training (True or False)
)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # use GPU if available


# get dataloaders
train_loader, validation_loader, test_loader, mean, std = get_dataloaders(
    batch_size=batch_size
)

# define sizes
input_dim = 1
hidden_dim = 1
output_dim = 1

# define model
model = RNNModel(input_dim, hidden_dim, output_dim).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss(reduction="sum")


# define variables to store loss and accuracy
train_history_loss = []
train_history_acc = []
val_history_loss = []
val_history_acc = []
test_history_loss = []
test_history_acc = []

for epoch in range(1, epochs + 1):
    model.train()

    train_loss = []
    train_acc = []

    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # place data in the correct device
        inputs = inputs.to(device)
        labels = labels.to(device).view(-1, 1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)

        # loss
        loss = loss_fn(outputs, labels)

        # backward
        loss.backward()

        # optimize
        optimizer.step()

        # keep loss for plotting
        train_loss.append(loss.item())

        # accuracy
        accuracy = (outputs.round) == labels

        # keep accuracy for plotting
        train_acc.append(accuracy)

    train_history_loss.append(np.mean(train_loss))
    train_history_acc.append(np.mean(train_acc))

    print(
        "[TRAIN]Epoch {:>3}/{:>3}, loss {:.4f}, acc {:.2f}".format(
            epoch, epochs, train_history_loss[-1], train_history_acc[-1]
        )
    )

    if epoch % eval_every == 0:
        val_loss, val_acc = evaluate(model, validation_loader, loss_fn, device)
        val_history_loss.append(val_loss)
        val_history_acc.append(val_acc)

        print(
            "[VAL]Epoch {:>3}/{:>3}, loss {:.4f}, acc {:.2f}".format(
                epoch, epochs, val_history_loss[-1], val_history_acc[-1]
            )
        )

        # if early stopping is implemented
        ...

        if evaluate_testset_during_training:
            test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
            test_history_loss.append(test_loss)
            test_history_acc.append(test_acc)

# Plotting loss for test and validation: simpleRNN
plt.figure(figsize=(10, 6))
plt.plot(train_history_loss, label="Training Loss")
plt.plot(val_history_loss, label="Validation Loss")

if evaluate_testset_during_training:
    plt.plot(test_history_loss, label="Test Loss")

plt.title("Loss History (Train, Validation, Test)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("simpleRNN.eps", format="eps")
plt.show()
plt.close()


# larger LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Define your new architecture here

        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Define the forward pass
        x = x.view(x.shape[0], x.shape[1], 1)
        out = self.rnn(x)[0]
        out = out[:, -1, :]
        out = self.output_layers(out)
        return out


# define model
model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss(reduction="sum")

# define variables to store loss and accuracy
train_history_loss = []
train_history_acc = []
val_history_loss = []
val_history_acc = []
test_history_loss = []
test_history_acc = []

for epoch in range(1, epochs + 1):
    model.train()

    train_loss = []
    train_acc = []

    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # place data in the correct device
        inputs = inputs.to(device)
        labels = labels.to(device).view(-1, 1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)

        # loss
        loss = loss_fn(outputs, labels)

        # backward
        loss.backward()

        # optimize
        optimizer.step()

        # keep loss for plotting
        train_loss.append(loss.item())

        # accuracy
        accuracy = (outputs.round) == labels

        # keep accuracy for plotting
        train_acc.append(accuracy)

    train_history_loss.append(np.mean(train_loss))
    train_history_acc.append(np.mean(train_acc))

    print(
        "[TRAIN]Epoch {:>3}/{:>3}, loss {:.4f}, acc {:.2f}".format(
            epoch, epochs, train_history_loss[-1], train_history_acc[-1]
        )
    )

    if epoch % eval_every == 0:
        val_loss, val_acc = evaluate(model, validation_loader, loss_fn, device)
        val_history_loss.append(val_loss)
        val_history_acc.append(val_acc)

        print(
            "[VAL]Epoch {:>3}/{:>3}, loss {:.4f}, acc {:.2f}".format(
                epoch, epochs, val_history_loss[-1], val_history_acc[-1]
            )
        )

        # if early stopping is implemented
        ...

        if evaluate_testset_during_training:
            test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
            test_history_loss.append(test_loss)
            test_history_acc.append(test_acc)

# Plotting loss for test and validation: LargerLSTM
plt.figure(figsize=(10, 6))
plt.plot(train_history_loss, label="Training Loss")
plt.plot(val_history_loss, label="Validation Loss")

if evaluate_testset_during_training:
    plt.plot(test_history_loss, label="Test Loss")

plt.title("Loss History (Train, Validation, Test)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("largerLSTM.eps", format="eps")
plt.show()
plt.close()


# GRU
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        # Define your new architecture here

        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Define the forward pass
        x = x.view(x.shape[0], x.shape[1], 1)
        out = self.rnn(x)[0]
        out = out[:, -1, :]
        out = self.output_layers(out)
        return out


# define model
model = GRUModel(input_dim, hidden_dim, output_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss(reduction="sum")

# define variables to store loss and accuracy
train_history_loss = []
train_history_acc = []
val_history_loss = []
val_history_acc = []
test_history_loss = []
test_history_acc = []

for epoch in range(1, epochs + 1):
    model.train()

    train_loss = []
    train_acc = []

    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # place data in the correct device
        inputs = inputs.to(device)
        labels = labels.to(device).view(-1, 1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)

        # loss
        loss = loss_fn(outputs, labels)

        # backward
        loss.backward()

        # optimize
        optimizer.step()

        # keep loss for plotting
        train_loss.append(loss.item())

        # accuracy
        accuracy = (outputs.round) == labels

        # keep accuracy for plotting
        train_acc.append(accuracy)

    train_history_loss.append(np.mean(train_loss))
    train_history_acc.append(np.mean(train_acc))

    print(
        "[TRAIN]Epoch {:>3}/{:>3}, loss {:.4f}, acc {:.2f}".format(
            epoch, epochs, train_history_loss[-1], train_history_acc[-1]
        )
    )

    if epoch % eval_every == 0:
        val_loss, val_acc = evaluate(model, validation_loader, loss_fn, device)
        val_history_loss.append(val_loss)
        val_history_acc.append(val_acc)

        print(
            "[VAL]Epoch {:>3}/{:>3}, loss {:.4f}, acc {:.2f}".format(
                epoch, epochs, val_history_loss[-1], val_history_acc[-1]
            )
        )

        # if early stopping is implemented
        ...

        if evaluate_testset_during_training:
            test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
            test_history_loss.append(test_loss)
            test_history_acc.append(test_acc)

# Plotting loss for test and validation: GRU
plt.figure(figsize=(10, 6))
plt.plot(train_history_loss, label="Training Loss")
plt.plot(val_history_loss, label="Validation Loss")

if evaluate_testset_during_training:
    plt.plot(test_history_loss, label="Test Loss")

plt.title("Loss History (Train, Validation, Test)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
file = "GRU.eps "
plt.savefig(file, format="eps")
plt.show()
plt.close()
print(f"Printed {file} to current directory.")

#
# Second part: predicting trajectories.
#


def get_dataloaders(batch_size, p=0.7):
    # Load dataset
    X_train = np.load(os.path.join("data", "X_train.npy"))
    y_train = np.load(os.path.join("data", "c_train.npy"))

    X_val = np.load(os.path.join("data", "X_val.npy"))
    y_val = np.load(os.path.join("data", "c_val.npy"))

    X_test = np.load(os.path.join("data", "X_test.npy"))
    y_test = np.load(os.path.join("data", "c_test.npy"))

    # normalization of the data
    mean_x = X_train.mean(0)
    std_x = X_train.std(0)
    X_train = (X_train - mean_x) / std_x
    X_test = (X_test - mean_x) / std_x
    X_val = (X_val - mean_x) / std_x

    mean_c = y_train.mean(0)
    std_c = y_train.std(0)
    y_train = (y_train - mean_c) / std_c
    y_test = (y_test - mean_c) / std_c
    y_val = (y_val - mean_c) / std_c

    train_loader = DataLoader(
        dataset=TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        dataset=TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, validation_loader, test_loader, mean_x, mean_c, std_x, std_c


(
    train_loader,
    validation_loader,
    test_loader,
    mean_x,
    mean_y,
    std_x,
    std_y,
) = get_dataloaders(batch_size=64)


class RNNModelGen(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModelGen, self).__init__()
        # Define your new architecture here
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, c, teacher_forcing=False):
        # Define the forward pass
        sequence_length = x.shape[1]

        # initialize the memory unit from the clinical data
        hn = self.input_layer(c)
        hn = hn.view(1, hn.shape[0], hn.shape[1])

        generated_sequence = []

        for i in range(sequence_length - 1):
            if teacher_forcing or i == 0:
                # Teacher forcing: Feed the target as the next input
                input_x = x[:, i].view(-1, 1, 1)
                if i == 0:
                    generated_sequence.append(input_x.view(-1))
            else:
                # Without teacher forcing: use the model's own predictions
                input_x = predicted_value

            out, hn = self.rnn(input_x, hn)

            predicted_value = self.output_layer(out)
            generated_sequence.append(predicted_value.view(-1))

        generated_sequence = torch.stack(generated_sequence, dim=1)
        return generated_sequence


def evaluate(model, data_loader, loss_fn, device):
    losses = []

    model.eval()
    for i, data in enumerate(data_loader):
        # get the inputs; data is a list of [inputs, labels]
        X, clinical = data

        # place data in the correct device
        X = X.to(device)
        clinical = clinical.to(device)
        with torch.no_grad():
            # forward
            outputs = model(X, clinical)

            # loss
            loss = loss_fn(outputs, X)

            # keep loss for plotting
            losses.append(loss.item())

    return np.mean(losses)


np.random.seed(13)

# define hyperparameters
epochs = 10
batch_size = 10
learning_rate = 1e-3
eval_every = 1  # number of epochs between evaluations
evaluate_testset_during_training = (
    True  # whether to evaluate the testset during training (True or False)
)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # use GPU if available


# get dataloaders
(
    train_loader,
    validation_loader,
    test_loader,
    mean_x,
    mean_y,
    std_x,
    std_y,
) = get_dataloaders(batch_size=batch_size)

# define sizes
input_dim = 8
hidden_dim = 36
output_dim = 1

# define model
model = RNNModelGen(input_dim, hidden_dim, output_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()


# define variables to store loss and accuracy
train_history_loss = []
val_history_loss = []
test_history_loss = []

for epoch in range(1, epochs + 1):
    model.train()

    train_loss = []

    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        X, clinical = data

        # place data in the correct device
        X = X.to(device)
        clinical = clinical.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(X, clinical)

        # loss
        loss = model.loss(outputs, X)

        # backward
        loss.backward()

        # optimize
        optimizer.step()

        # keep loss for plotting
        losses.append(loss.item())

    train_history_loss.append(np.mean(train_loss))

    print(
        "[TRAIN]Epoch {:>3}/{:>3}, loss {:.4f}".format(
            epoch, epochs, train_history_loss[-1]
        )
    )

    if epoch % eval_every == 0:
        val_loss = evaluate(model, validation_loader, loss_fn, device)
        val_history_loss.append(val_loss)
        print(
            "[VAL]Epoch {:>3}/{:>3}, loss {:.4f}".format(
                epoch, epochs, val_history_loss[-1]
            )
        )

        # if early stopping is implemented
        ...

        if evaluate_testset_during_training:
            test_loss = evaluate(model, test_loader, loss_fn, device)
            test_history_loss.append(test_loss)


for loader in [train_loader, validation_loader, test_loader]:
    x, clinical = loader.dataset.tensors
    x = x.to(device)
    clinical = clinical.to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(x, clinical)

    predictions = predictions.cpu().numpy()
    x = x.cpu().numpy()

    predictions = (predictions * std_x) + mean_x
    x = (x * std_x) + mean_x

    t = list(range(x.shape[1]))

    mu_0 = predictions.mean(0)
    std_0 = predictions.std(0)

    mu_1 = x.mean(0)
    std_1 = x.std(0)

    plt.plot(t, mu_1, lw=2, label="True", color="#5387DD")
    plt.fill_between(t, mu_1 + std_1, mu_1 - std_1, facecolor="#5387DD", alpha=0.3)
    plt.plot(t, mu_0, lw=2, label="Prediction", color="#C565C7")
    plt.fill_between(t, mu_0 + std_0, mu_0 - std_0, facecolor="#C565C7", alpha=0.3)
    plt.legend()
    plt.savefig("predicted.eps", format="eps")
    plt.show()
    plt.close()


idx = 0

x, clinical = train_loader.dataset.tensors
x = x.to(device)
clinical = clinical.to(device)

model.eval()
with torch.no_grad():
    predictions = model(x, clinical)

predictions = predictions.cpu().numpy()
x = x.cpu().numpy()

predictions = (predictions * std_x) + mean_x
x = (x * std_x) + mean_x

plt.plot(predictions[idx], label="Prediction")
plt.plot(x[idx], label="True")
plt.legend()
plt.title("Prediction vs True for one sample")
plt.savefig("predictionVsTrue.eps", format="eps")
plt.show()
plt.close()
