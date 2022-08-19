import torch


def gram_matrix(y):
    """
    Calculates the Gram matrix of the input y
    """
    # (b, ch, h, w) = y.size()
    # features = y.view(b, ch, w * h)
    # features_t = features.transpose(1, 2)
    # gram = features.bmm(features_t) / (ch * h * w)
    # return gram
    b, c, h, w = y.size()
    y = y.view(b * c, h * w)
    gram_matrix = torch.mm(y, y.t()).div(b * c * h * w * 2)
    # gram_matrix = y
    return gram_matrix
