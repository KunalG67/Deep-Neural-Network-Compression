from .conv2d import modified_conv2d
from .linear import modified_linear


def quantize_model(model, k=16):
    """
    Applies k-means quantization to all modified layers.
    k = number of weight clusters (centroids).
    Paper uses k=2^b where b is number of bits (e.g. k=16 for 4-bit).
    """
    for module in model.modules():
        if isinstance(module, (modified_conv2d, modified_linear)):
            module.quantize(k)
    print(f"[Quantization] Done — k={k} centroids ({k.bit_length()-1}-bit)")