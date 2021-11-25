from image_dataset import ImageData
from torch.utils.data import DataLoader
import torch
from torch import nn
from model_setr_pup import SETR_PUP
from model_setr_mla import SETR_MLA


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

data_train = ImageData('train')
data_validation = ImageData('validation')

batch_size = 4
learning_rate = 1e-5
epochs = 500

train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(data_validation, batch_size=batch_size, shuffle=True)

# model = SETR_PUP().to(device)
model = SETR_MLA().to(device)


loss_cross_entropy = nn.CrossEntropyLoss()
def loss_fn(pred, y):

    batch, channel, height, width = pred.shape
    loss = loss_cross_entropy(pred, y)
    
    return loss

def accuracy_fn(pred, y):
    accuracy = (pred.argmax(1) == y)

    batch, height, width = accuracy.shape

    accuracy = accuracy.type(torch.float)
    accuracy = accuracy.reshape(batch, height*width)
    accuracy = accuracy.mean(dim=-1)
    accuracy = accuracy.sum()

    return accuracy

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss

        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += accuracy_fn(pred, y).item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct

min_test_loss = 1e10
best_accuracy = 0

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_accuracy = test_loop(validation_dataloader, model, loss_fn)

    if test_loss < min_test_loss:
        min_test_loss = test_loss
        best_accuracy = test_accuracy

        print("best accuracy: {}, min. loss: {}".format(best_accuracy, min_test_loss))
        torch.save(model.state_dict(), 'best_weight_setr_pup.pth')

print("best accuracy: {}, min. loss: {}".format(best_accuracy, min_test_loss))

print("Done!")
