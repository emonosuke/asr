def frame_stacking(x, num_framestack):
    newlen = x.shape[0] // num_framestack
    lmfb_dim = x.shape[1]
    x_stacked = x[0:newlen * num_framestack].reshape(newlen, lmfb_dim * num_framestack)

    return x_stacked
