from torch import nn


class AcousticModel(nn.Module):
    """
    AcousticModel specifies the common attributes/methods that
    will be exposed by all Snowfall acoustic model networks.
    Think of it as of an interface class.
    """

    # A.k.a. the input feature dimension.
    num_features: int

    # A.k.a. the output dimension (could be the number of phones or
    # characters in the vocabulary).
    num_classes: int

    # When greater than one, the networks output sequence length will be
    # this many times smaller than the input sequence length.
    subsampling_factor: int
