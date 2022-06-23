import torch.nn as nn
import sys
import  snip_paper_functions as snip_implementations
from networks.vgg import vgg16
from data.dataprep import DataPrep

class Args:
    dataset='cifar10'
    epochs=180
    batch_size = 128
    criterion=nn.CrossEntropyLoss()
    sparsity_level = 0.15
    workers=3

global args
args = Args

def main():

    #Initialize a vgg 10 model
    model = vgg16(10)

    _dataprep = DataPrep(args.dataset, args)

    # Extract train and test loader

    train_loader, test_loader = _dataprep.get_loaders()

    # Before applying pruning by masking

    print("Absolute sparsity percentage and total nonzero params before masking is {1}", snip_implementations.get_model_sps(model))

    keep_masks = snip_implementations.get_scores_for_loader(model, train_loader, args.criterion, args.sparsity_level)

    masked_final_model = snip_implementations.apply_mask_to_model(model, keep_masks)

    # Before applying pruning by masking
    print("Absolute sparsity percentage and total nonzero params after masking is {1}",
          snip_implementations.get_model_sps(masked_final_model))

    # Normal training process

    # trainer.training(masked_final_model,train_loader, test_loader,self.criterion, optim.SGD(params=masked_final_model.parameters(),lr=0.0001),5)

    sys.exit()


if __name__ == '__main__':
       main()
