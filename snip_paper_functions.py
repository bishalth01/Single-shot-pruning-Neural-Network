import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch
import torch.nn as nn
import types
import torch.nn.functional as F
import copy



def snip_forward_linear(params, x):

    return F.linear(x, params.weight * params.weight_mask, params.bias)

def snip_forward_conv2d(params, x):

    return F.conv2d(x, params.weight * params.weight_mask, params.bias,
                        params.stride, params.padding, params.dilation, params.groups)


def get_scores_for_loader(net, train_dataloader, criterion, sparsity_level):
    device = next(iter(net.parameters())).device
    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Creating deep copy to make sure we don't modify gradient requirement for training

    net = copy.deepcopy(net)

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight)).to(device)
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)


    net.to(device)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    grads_abs = []

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads_abs.append(torch.abs(layer.weight_mask.grad))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * sparsity_level)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    initialize_mult = []
    for i in range(len(grads_abs)):
        initialize_mult.append(grads_abs[i] / norm_factor)
    return keep_masks


def apply_mask_to_model(model, mask):
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),
        model.modules()
    )
    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in mask])))
    total_non_zero = 0
    for layer, keep_mask in zip(prunable_layers, mask):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        total_non_zero += torch.sum(torch.cat([torch.flatten(x != 0.) for x in layer.weight.data]))
        layer.weight.register_hook(hook_factory(keep_mask))

    return model


def get_model_sps(model):
    nonzero = total = 0
    for name, param in model.named_parameters():
        if 'mask' not in name:
            nz_count = torch.count_nonzero(param).item()
            total_params = param.numel()
            nonzero += nz_count
            total += total_params

    abs_sps = 100 * (total - nonzero) / total

    return abs_sps, nonzero