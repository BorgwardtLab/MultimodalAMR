def calc_conv1d_output_size(input_size, kernel_sizes, stride=1, padding=0, dilation=1):
    size = input_size
    for ks in kernel_sizes:
        size = (size + 2 * padding - dilation * (ks - 1) - 1) / stride + 1
    return size
