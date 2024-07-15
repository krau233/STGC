import torch
def to_one_hot(targets, num_classes):
    one_hot = torch.zeros(len(targets), num_classes,dtype=int)
    one_hot.scatter_(1, targets.unsqueeze(1), 1)
    return one_hot