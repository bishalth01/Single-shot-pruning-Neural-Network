import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch
import torch.nn as nn
import  snip_paper_functions as snip_implementations

class Args:
    epochs=180
    batch_size = 128
    random_state = 42
    test_size=0.25
    batch_size=25
    num_workers=1
    shuffle=True
    drop_last=True
    sparsity_level=0.15
    criterion=nn.MSELoss()

global args
args = Args

def main():
    torch.manual_seed(1)
    # Initializing a custom MLP
    model = nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 64),
                          nn.Linear(64, 1))

    # Initializing sample data
    torch.manual_seed(2)
    X = torch.randn(500, 100)
    torch.manual_seed(3)
    y = torch.randn(500)
    torch.manual_seed(4)

    # Simple test train split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    # Creating Loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle,
                                               drop_last=args.drop_last)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle,drop_last=args.drop_last)

    # Before applying pruning by masking

    print("Absolute sparsity percentage and total nonzero params before pruning is {1}", snip_implementations.get_model_sps(model))

    keep_masks = snip_implementations.get_scores_for_loader(model, train_loader, args.criterion, args.sparsity_level)

    masked_final_model = snip_implementations.apply_mask_to_model(model, keep_masks)

    # Before applying pruning by masking
    print("Absolute sparsity percentage and total nonzero params after pruning is {1}",snip_implementations.get_model_sps(masked_final_model))

    # Normal training process

    # trainer.training(masked_final_model,train_loader, test_loader,self.criterion, optim.SGD(params=masked_final_model.parameters(),lr=0.0001),5)


if __name__ == '__main__':
        main();

