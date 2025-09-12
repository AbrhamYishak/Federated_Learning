import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

NUM_CLIENTS   = 4
ROUNDS        = 3
LR            = 0.01
MOMENTUM      = 0.9
LOCAL_EPOCHS  = 1
BATCH_SIZE    = 32
CLIENT_FRAC   = 1.0

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

def get_testloader(batch_size=32):
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

def iid_partition(dataset, num_clients):
    num_items = int(len(dataset)/num_clients)
    all_indices = np.arange(len(dataset))
    np.random.shuffle(all_indices)
    return {i: all_indices[i*num_items:(i+1)*num_items] for i in range(num_clients)}

def noniid_partition(dataset, num_clients, shards_per_client=2):
    num_shards = num_clients * shards_per_client
    shard_size = len(dataset) // num_shards
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    shards = [idxs[i*shard_size:(i+1)*shard_size] for i in range(num_shards)]
    np.random.shuffle(shards)

    client_dict = {i: np.array([], dtype=int) for i in range(num_clients)}
    for i in range(num_clients):
        chosen = shards[shards_per_client*i : shards_per_client*(i+1)]
        client_dict[i] = np.concatenate(chosen, axis=0)
    return client_dict

iid_data = iid_partition(trainset, NUM_CLIENTS)
noniid_data = noniid_partition(trainset, NUM_CLIENTS)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

def train(model, trainloader, epochs=1):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.NLLLoss()
    model.train()
    for _ in range(epochs):
        for data, target in trainloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def test(model, testloader):
    criterion = nn.NLLLoss()
    correct, loss_total = 0, 0
    model.eval()
    with torch.no_grad():
        for data, target in testloader:
            output = model(data)
            loss_total += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss_total / len(testloader), accuracy

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data_indices, batch_size, local_epochs):
        self.cid = cid
        self.model = Net()
        self.trainloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset, data_indices),
            batch_size=batch_size, shuffle=True
        )
        self.local_epochs = local_epochs

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=self.local_epochs)
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, get_testloader(BATCH_SIZE))
        return float(loss), len(testset), {"accuracy": float(accuracy)}

def run_simulation(partitioned_data, num_rounds, batch_size, local_epochs, client_frac):
    acc_list = []

    def client_fn(cid: str):
        return FlowerClient(cid, partitioned_data[int(cid)], batch_size, local_epochs)

    def evaluate(server_round, parameters, config):
        model = Net()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, get_testloader(batch_size))
        acc_list.append(accuracy)  # 👈 Manual storage
        return loss, {"accuracy": accuracy}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=client_frac,
        evaluate_fn=evaluate
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(partitioned_data),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )
    return acc_list

iid_acc = run_simulation(iid_data, ROUNDS, BATCH_SIZE, LOCAL_EPOCHS, CLIENT_FRAC)
noniid_acc = run_simulation(noniid_data, ROUNDS, BATCH_SIZE, LOCAL_EPOCHS, CLIENT_FRAC)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(iid_acc)+1), iid_acc, label="IID")
plt.plot(range(1, len(noniid_acc)+1), noniid_acc, label="Non-IID")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("IID vs Non-IID Federated Learning (MNIST)")
plt.legend()
plt.show()

results = {}

# Vary E
for e in [1, 3]:
    results[f"E={e}"] = run_simulation(iid_data, ROUNDS, BATCH_SIZE, e, CLIENT_FRAC)

# Vary B
for b in [16, 64]:
    results[f"B={b}"] = run_simulation(iid_data, ROUNDS, b, LOCAL_EPOCHS, CLIENT_FRAC)

# Vary C
for c in [0.5, 1.0]:
    results[f"C={c}"] = run_simulation(iid_data, ROUNDS, BATCH_SIZE, LOCAL_EPOCHS, c)

plt.figure(figsize=(8,5))
for label, acc in results.items():
    plt.plot(range(1, len(acc)+1), acc, label=label)
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Effect of E, B, C on Federated Learning (MNIST, IID)")
plt.legend()
plt.show()
