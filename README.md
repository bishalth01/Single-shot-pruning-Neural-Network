# Single-shot-pruning-Neural-Network

This is the implementation of SNIP: Single-shot Network Pruning based on Connection Sensitivity (https://arxiv.org/abs/1810.02340). The main idea is based on connection sensitivity that identifies important connections in the network before training, creating a sparse network for training.

main.py - Main function with custom tensor data (MLP).

main_with_vgg - Main function with cifar-10 dataset on VGG 16 network.

Referenced from : https://github.com/riohib/distributed-pruning
