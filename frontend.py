import random


def frame_stacking(x, num_framestack):
    newlen = x.shape[0] // num_framestack
    lmfb_dim = x.shape[1]
    x_stacked = x[0:newlen * num_framestack].reshape(newlen, lmfb_dim * num_framestack)

    return x_stacked


def spec_augment(x, lmfb_dim, max_mask_freq, max_mask_time):
    """ frequency masking and time masking
    TODO: time warping
    """
    mask_freq = random.randint(0, max_mask_freq)
    mask_freq_from = random.randint(0, lmfb_dim - mask_freq)
    mask_freq_to = mask_freq_from + mask_freq
    x[:, mask_freq_from:mask_freq_to] = 0.0

    len_t = x.shape[0]
    if len_t > max_mask_time:
        mask_time = random.randint(0, max_mask_time)
        mask_time_from = random.randint(0, len_t - mask_time)
        mask_time_to = mask_time_from + mask_time
        x[mask_time_from:mask_time_to, :] = 0.0
    return x
