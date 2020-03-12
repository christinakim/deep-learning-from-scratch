import torch
from torch import nn
from torch.optim import Adam
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def training_model_once(model: nn.Module, train_data_loader: DataLoader, optimizer: Optimizer):
    """Returns training losses for one epoch while training the model"""
    model.train()
    training_losses = []
    for batch in train_data_loader:
        batch = batch.cuda()
        outputs = model.forward(batch)
        loss = model.loss(outputs, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())
    return training_losses


def get_validation_loss(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for minibatch in data_loader:
            minibatch = minibatch.cuda()
            outputs = model.forward(minibatch)
            loss = model.loss(outputs, minibatch)
            total_loss += loss * minibatch.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss.item()


def training_loop(model, train_data_loader, test_data_loader, epochs=10, lr=1e-3, optimizer=Adam, silent=False):
    optimizer = optimizer(model.parameters(), lr=lr)
    train_losses = []
    test_losses = [get_validation_loss(model, test_data_loader)]
    for epoch in range(epochs):
        train_losses.extend(training_model_once(model, train_data_loader, optimizer))
        test_loss = get_validation_loss(model, test_data_loader)
        test_losses.append(test_loss)
        if not silent:
            print(f"Epoch {epoch}, Test loss {test_loss:.4f}")
    return train_losses, test_losses


def debug_grad(module, grad_input, grad_output):
    for grad in grad_input:
        assert torch.isfinite(grad).all()
    for grad in grad_output:
        assert torch.isfinite(grad).all()


def get_toy_dataset():
    return torch.tensor(
        [
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 1),
        ]
        * 10
    )


def get_paramed_dataset(topright, topleft, bottomright, bottomleft):
    return tensor()
