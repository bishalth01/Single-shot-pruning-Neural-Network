import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch


def train_model(model, train_dataloader, validation_dataloader, criterion, optimizer):

    train_loss = 0
    validation_loss = 0
    model.train()

    for local_batch, local_labels in train_dataloader:
            X = local_batch
            y = local_labels

            # Calculate output
            output = model(X)
            loss = criterion(output, y)

            # Calculate gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Update train loss

            train_loss += loss.item()
    else:
        model.eval()
        with torch.no_grad():
            for local_batch, local_labels in validation_dataloader:
                # compute output
                output = model(local_batch)
                loss = criterion(output, local_labels)

                output = output.float()
                loss = loss.float()

                validation_loss += loss.item()

    return train_loss, validation_loss

def training(model, train_dataloader, validation_dataloader, criterion, optimizer, epochs):


    train_losses, validation_losses=[],[]
    for e in range(1, epochs + 1):
        train_loss_for_all_batch, validation_losses_for_all_batch = train_model(model, train_dataloader,validation_dataloader, criterion, optimizer)
        train_losses.append(train_loss_for_all_batch/len(train_dataloader))
        validation_losses.append(validation_losses_for_all_batch / len(validation_dataloader))



        print("Epoch {}/{}".format(e, epochs),
              "train loss = {:.5f}".format(train_losses),
              "validation loss = {:.5f}".format(validation_losses))
        print("Loss recorded is ", abs(validation_losses-train_losses))















