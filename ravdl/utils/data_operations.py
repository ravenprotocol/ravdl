import ravop as R
import math

def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    equality = R.equal(y_true, y_pred)
    accuracy = R.div(R.sum(equality, axis=0), R.index(R.shape(y_true),indices='[0]'))
    return accuracy

def determine_padding(filter_shape, output_shape="same"):
    # No padding
    if output_shape == "valid":
        return (0, 0), (0, 0)
    # Pad so that the output shape is the same as input shape (given that stride=1)
    elif output_shape == "same":
        filter_height, filter_width = filter_shape

        # Derived from:
        # output_height = (height + pad_h - filter_height) / stride + 1
        # In this case output_height = height and stride = 1. This gives the
        # expression for the padding below.
        pad_h1 = int(math.floor((filter_height - 1)/2))
        pad_h2 = int(math.ceil((filter_height - 1)/2))
        pad_w1 = int(math.floor((filter_width - 1)/2))
        pad_w2 = int(math.ceil((filter_width - 1)/2))

        return (pad_h1, pad_h2), (pad_w1, pad_w2)