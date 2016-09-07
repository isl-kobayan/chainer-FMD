import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class SwitchChannel(function.Function):

    """Switch outputs of each channel."""

    def __init__(self, switch):
        self.switch = switch.reshape((1,-1,1,1))

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

        x_type = in_types[0]
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 4,
            x_type.shape[1] == self.switch.shape[1]
        )

    def forward(self, inputs):
        x = inputs[0]
        xp = cuda.get_array_module(x)
        if xp == numpy:
            self.mask = switch
        else:
            self.mask = cuda.to_gpu(self.switch)
        return x * self.mask,

    def backward(self, x, gy):
        return gy[0] * self.mask,


def switch_channel(x, switch=None):
    """Switches filter activation.
    This function switches filter activation.
    Args:
        x (~chainer.Variable): Input variable.
        switch (numpy.ndarray): switch flags. Each elements should be 0 or 1.
    Returns:
        ~chainer.Variable: Output variable.
    """
    if switch is not None:
        return SwitchChannel(switch)(x)
    return x
