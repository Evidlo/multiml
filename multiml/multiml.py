#!/usr/bin/env python

import numpy as np
from tqdm import tqdm
from skimage.transform import rescale, downscale_local_mean

def correlate_and_sum(frames, mode='CC', disable_print=True, np=np):
    """Correlate all frame combinations and sum each group of correlations

    Args:
        frames (ndarray): input images
        mode (str, default='PC'): type of correlation to use. ('PC', 'NCC', 'CC')

    Returns:
        ndarray: axes (group, corr_x_coord, corr_y_coord)
    """

    image_axes = range(1, len(frames.shape))
    frames_freq = np.fft.fftn(frames, axes=image_axes)

    product_sums = np.zeros(
        (len(frames) - 1, *frames.shape[1:]),
        dtype='complex128'
    )
    for time_diff in tqdm(
            range(1, len(frames_freq)), desc='Correlation', leave=None,
            disable=disable_print
    ):
        products = frames_freq[:-time_diff] * frames_freq[time_diff:].conj()
        if mode.upper() == 'PC':
            product_sums[time_diff - 1] = np.sum(products / np.abs(products), axis=0)
        elif mode.upper() == 'CC':
            product_sums[time_diff - 1] = np.sum(products, axis=0)
        else:
            raise Exception('Invalid mode {}'.format(mode.upper()))

    return np.fft.ifftn(np.array(product_sums), axes=image_axes).real


def scale_and_sum(csums, scale_dir='down', disable_print=True):
    """Scale each correlation group so peaks are incident, then crop and sum

    Args:
        csums (ndarray): correlation sums
        scale_dir (str): 'up' or 'down'
        disable_print (boolean): disable tqdm printing

    Returns:
        result (ndarray): summed result
        scaled_csums (ndarray): scaled and cropped version of correlation groups

    """

    # shape_x, shape_y = csums.shape[1], csums.shape[2]
    # result = np.zeros((shape_x, shape_y))
    output_shape = csums.shape[1:]
    result = np.zeros((output_shape))
    scaled_csums = []

    for diff, csum in enumerate(
            tqdm(csums, desc='Scale', leave=None, disable=True),
            1
    ):
        # pretrim csum to relevant area to make scaling faster
        trim_slice = tuple([slice(None, int(shape / (len(csums) / diff) + 1)) for shape in output_shape])
        scaled = csum[trim_slice]
        # print('precrop', scaled.shape)
        # scale correlation group so peaks are incident
        if scale_dir == 'up':
            scaled = rescale(csum, len(csums) / diff, anti_aliasing=False)
        else:
            # scaled = downscale_local_mean(csum, (diff, diff))
            scale_slice = tuple([slice(None, None, diff)] * len(csums.shape[1:]))
            scaled = csum[scale_slice]
            pad_slice = tuple([slice(None, shape) for shape in scaled.shape])
            padded = np.zeros((output_shape))
            padded[pad_slice] = scaled
            # scaled = size_equalizer(scaled, output_shape, mode='topleft')
            scaled = padded

        # print('scale', len(csums) / diff)
        # trim off excess
        trim_slice = tuple([slice(None, shape) for shape in output_shape])
        scaled = scaled[trim_slice]
        # print('crop')
        scaled_csums.append(scaled)
        result += scaled

    return result, np.array(scaled_csums)


def register(frames, scale_dir='down', disable_print=True):
    """Register frames with Multi ML method

    Args:
        frames (ndarray): stack of frames to register
        scale_dir (str): scale correlation groups 'up' (subpixel registration) or
            'down'
        disable_print (boolean): disable printing
    """
    csums = correlate_and_sum(frames, disable_print=disable_print)
    result, scaled = scale_and_sum(csums, scale_dir=scale_dir, disable_print=disable_print)

    return np.array(np.unravel_index(
        np.argmax(result),
        result.shape))

    # # region to search for argmax assuming cK < N
    # search_region = np.array(frames[0].shape) // len(frames) + (1, 1)

    # return np.array(np.unravel_index(
    #     np.argmax(result[:search_region[0], :search_region[1]]),
    #     search_region))

def shift_and_sum(frames, drift, mode='full', shift_method='roll'):
    """Coadd frames by given shift

    Args:
        frames (ndarray): input frames to coadd
        drift (ndarray): drift between adjacent frames
        mode (str): zeropad before coadding ('full') or crop to region of
            frame overlap ('crop'), or crop to region of first frame ('first')
        shift_method (str): method for shifting frames ('roll', 'fourier')
        pad (bool): zeropad images before coadding

    Returns:
        (ndarray): coadded images
    """

    assert type(drift) is np.ndarray, "'drift' should be ndarray"

    pad = np.ceil(drift * (len(frames) - 1)).astype(int)
    pad_r = (0, pad[0]) if drift[0] > 0 else (-pad[0], 0)
    pad_c = (0, pad[1]) if drift[1] > 0 else (-pad[1], 0)
    frames_ones = np.pad(
        np.ones(frames.shape, dtype=int),
        ((0, 0), pad_r, pad_c),
        mode='constant',
    )
    frames_pad = np.pad(frames, ((0, 0), pad_r, pad_c), mode='constant')

    summation = np.zeros(frames_pad[0].shape, dtype='complex128')
    summation_scale = np.zeros(frames_pad[0].shape, dtype=int)

    for time_diff, (frame, frame_ones) in enumerate(zip(frames_pad, frames_ones)):
        shift = np.array(drift) * (time_diff + 1)
        if shift_method == 'roll':
            integer_shift = np.floor(shift).astype(int)
            shifted = roll(frame, (integer_shift[0], integer_shift[1]))
            shifted_ones = roll(frame_ones, (integer_shift[0], integer_shift[1]))
        elif shift_method == 'fourier':
            shifted = np.fft.ifftn(fourier_shift(
                np.fft.fftn(frame),
                (shift[0], shift[1])
            ))
            shifted_ones = np.fft.ifftn(fourier_shift(
                np.fft.fftn(frame_ones),
                (shift[0], shift[1])
            ))
        else:
            raise Exception('Invalid shift_method')
        summation += shifted
        summation_scale += shifted_ones

    summation /= len(frames)

    if mode == 'crop':
        summation = size_equalizer(
            summation,
            np.array(frames_pad[0].shape).astype(int) -
            np.ceil(drift * (len(frames_pad)-1)).astype(int)
        )
    elif mode == 'full':
        # FIXME: something wrong with full mode
        pass
    elif mode == 'first':
        summation_scale[summation_scale == 0] = 1
        summation = summation[:frames.shape[1], :frames.shape[2]]
    elif mode == 'center':
        summation_scale[summation_scale == 0] = 1
        summation = size_equalizer(summation, frames.shape[1:])
    else:
        raise Exception('Invalid mode')

    return summation.real

def roll(x, shift):
    shift = np.round(shift).astype(int)
    return np.roll(
        np.roll(
            x,
            shift[0],
            axis=0
        ),
        shift[1],
        axis=1
    )

def size_equalizer(x, ref_size, mode='center'):
    """
    Crop or zero-pad a 2D array so that it has the size `ref_size`.
    Both cropping and zero-padding are done such that the symmetry of the
    input signal is preserved.
    Args:
        x (ndarray): array which will be cropped/zero-padded
        ref_size (list): list containing the desired size of the array [r1,r2]
        mode (str): ('center', 'topleft') where x should be placed when zero padding
    Returns:
        ndarray that is the cropper/zero-padded version of the input
    """
    assert len(x.shape) == 2, "invalid shape for x"

    if x.shape[0] > ref_size[0]:
        pad_left, pad_right = 0, 0
        crop_left = 0 if mode == 'topleft' else (x.shape[0] - ref_size[0] + 1) // 2
        crop_right = crop_left + ref_size[0]
    else:
        crop_left, crop_right = 0, x.shape[0]
        pad_right = ref_size[0] - x.shape[0] if mode == 'topleft' else (ref_size[0] - x.shape[0]) // 2
        pad_left = ref_size[0] - pad_right - x.shape[0]
    if x.shape[1] > ref_size[1]:
        pad_top, pad_bottom = 0, 0
        crop_top = 0 if mode == 'topleft' else (x.shape[1] - ref_size[1] + 1) // 2
        crop_bottom = crop_top + ref_size[1]
    else:
        crop_top, crop_bottom = 0, x.shape[1]
        pad_bottom = ref_size[1] - x.shape[1] if mode == 'topleft' else (ref_size[1] - x.shape[1]) // 2
        pad_top = ref_size[1] - pad_bottom - x.shape[1]

    # crop x
    cropped = x[crop_left:crop_right, crop_top:crop_bottom]
    # pad x
    padded = np.pad(
        cropped,
        ((pad_left, pad_right), (pad_top, pad_bottom)),
        mode='constant'
    )

    return padded
