import model as model_py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


# Write Data loaders, training procedure and validation procedure in this file.


def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    epoch_loss = []
    for cur_step, sample_batched in enumerate(train_loader):
        data, target = sample_batched['input'].to(device), sample_batched['target'].to(device)
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backprop
        loss.backward()
        optimizer.step()

        # Stats
        epoch_loss.append(loss.item())
        if cur_step % 50 == 0:
            global_step = (epoch - 1) * len(train_loader) + cur_step
            print('step#', global_step, 'training loss', np.asarray(epoch_loss).mean())

    return np.asarray(epoch_loss).mean()


def val(model, val_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        epoch_loss = []
        for i, sample_batched in enumerate(val_loader):
            data, target = sample_batched['input'].to(device), sample_batched['target'].to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Stats
            epoch_loss.append(loss.item())

    return np.asarray(epoch_loss).mean()


def main():
    dataset = model_py.RGBDataset()
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.01, 0.002, 0.988], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = model_py.Model()

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    device = torch.device('cuda')
    model.cuda()

    epoch = 1
    best_loss = float('inf')
    max_epochs = 10
    while epoch <= max_epochs:
        print('Start epoch', epoch)
        train_loss = train(model, train_loader, criterion, optimizer, epoch, device)
        val_loss = val(model, val_loader, criterion, device)
        lr_scheduler.step(val_loss)
        print('Epoch (', epoch, '/', max_epochs, ')')
        print('---------------------------------')
        print('Train loss: %0.4f' % (train_loss))
        print('Val loss: %0.4f' % (val_loss))
        print('---------------------------------')
        # Save checkpoint if is best
        if epoch % 5 == 0 and val_loss < best_loss:
            best_loss = val_loss
            state = {'model_state_dict': model.state_dict(),
                     'epoch': epoch,
                     'model_loss': val_loss, }
            torch.save(state, 'bst.ckpt')
            print("checkpoint saved at epoch", epoch)
        epoch += 1


if __name__ == "__main__":
    main()
