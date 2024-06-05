import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from types import SimpleNamespace

from models.sparse import SparseMLP
from models.vit import FeedForward
from tqdm import tqdm 

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output

class Net(nn.Module):

    def __init__(self, model_dim, num_mlp = 3, sparse_heads = 8, mlp_mult = 1.0, dropout = 0., sparse=False, perm=False):
        super().__init__()
        self.proj_in = nn.Linear(28*28, model_dim)
        if sparse:
            self.mlps = nn.ModuleList([SparseMLP(dim=model_dim//sparse_heads, heads=sparse_heads, mlp_dim=int(model_dim * mlp_mult)//sparse_heads, dropout=dropout, perm=perm) for _ in range(num_mlp)])
        else:
            self.mlps = nn.ModuleList([FeedForward(dim=model_dim, hidden_dim=int(model_dim * mlp_mult), dropout=dropout) for _ in range(num_mlp)])
        self.norms = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(num_mlp)])
        self.proj_out = nn.Linear(model_dim, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.proj_in(x)
        for mlp, norm in zip(self.mlps, self.norms):
            x = mlp(norm(x))
        x = self.proj_out(x)
        return F.log_softmax(x, dim=-1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))
            pbar.set_description(f"Train Epoch: {epoch} Loss: {loss.item()}")
            pbar.update(len(data))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    args = dict(
        batch_size=64,
        test_batch_size=1000,
        epochs=14,
        lr=1.0,
        gamma=0.7,
        no_cuda=False,
        no_mps=False,
        dry_run=False,
        seed=1,
        log_interval=10,
        save_model=False,
        dim = 256,
        sparse_heads = 4,
        sparse = True,
        num_mlp = 2,
        mlp_mult = 1.0,
        compile = True
    )
    args = SimpleNamespace(**args)
    # torch.set_float32_matmul_precision("high")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net(args.dim, args.num_mlp, args.sparse_heads, args.mlp_mult, sparse=args.sparse).to(device)
    print(model)
    # import pdb; pdb.set_trace()
    if args.compile:
        model = torch.compile(model)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
