import numpy as np
from skimage.transform import downscale_local_mean, SimilarityTransform, warp
from tqdm import tqdm

def get_frames(
    *,
    scene,
    resolution_ratio,
    num_frames,
    drift,
    jitter_std,
    rotation_std,
    frame_size,
    start=(0, 0),
    silent=False,
    **kwargs
):
    """
    Get noisy observation frames

    Args:
        scene (ndarray): high-resolution input scene
        resolution_ratio (int): downsample factor from high-resolution
            scene to low-resolution frames
        num_frames (int): number of experiment frames
        drift (tuple): inter-frame drift, high-resolution grid
        jitter_std (float): std. dev of Normal dist. to sample 2D jitter from
        rotation_std (float): std. dev of Normal dist. to sample rotation jitter from (degrees)
        frame_size (:obj:`list` of :obj:`int`): pixels of size of square frame
        start (tuple): top-left coordinate of first frame
        silent (boolean): silence tqdm progress bar
    """

    # FIXME check box bounds correctly, need to account for rotation
    assert (
        start[0] >= 0 and start[1] >= 0 and
        start[0] + num_frames * drift[0] + frame_size[0] < scene.shape[0] and
        start[1] + num_frames * drift[1] + frame_size[1] < scene.shape[1]
    ), "Frames drift outside of scene bounds"

    # drift should be integer amount on HR grid
    assert(type(drift[0]) in (np.int64, int) and type(drift[1]) in (np.int64, int))

    # initialize output array
    frames = np.zeros((num_frames, frame_size[0], frame_size[1]))

    for frame_num in tqdm(range(num_frames), desc='Frames', leave=None, position=1, disable=silent):
        top = start[0] + frame_num * drift[0]
        left = start[1] + frame_num * drift[1]
        linear_jitter = np.random.normal(0, scale=jitter_std, size=2)
        rotation_jitter = np.random.normal(0, scale=rotation_std)
        # jittered_scene = np.fft.ifftn(fourier_shift(np.fft.fftn(scene), jitter))
        center = np.array((top + frame_size[0] // 2, left + frame_size[0] // 2))
        transform = (
            SimilarityTransform(translation=-center) +
            SimilarityTransform(rotation=np.deg2rad(np.random.choice((0, 1)))) +
            SimilarityTransform(translation=center + linear_jitter)
        )
        jittered_scene = warp(scene, transform)
        frames[frame_num] = downscale_local_mean(
            jittered_scene[
                top:top + frame_size[0] * resolution_ratio,
                left:left + frame_size[1] * resolution_ratio
            ],
            (resolution_ratio, resolution_ratio)
        )

    return frames

def add_noise(frames, noise_model=None, max_count=None, dbsnr=0, **kwargs):
    """
    Add noise to frames

    Args:
        noise_model (str): 'poisson' or 'gaussian'
        dbsnr (float): target SNR in db
    """

    if noise_model == 'gaussian':
        var_sig = np.var(frames)
        var_noise = var_sig / 10**(dbsnr / 10)
        out = np.random.normal(loc=frames, scale=np.sqrt(var_noise))
    elif noise_model == 'poisson':
        if max_count is not None:
            sig_scaled = frames * (max_count / frames.max())
            # print('SNR:{}'.format(np.sqrt(sig_scaled.mean())))
            out = np.random.poisson(sig_scaled) * (frames.max() / max_count)
        else:
            avg_brightness = 10**(dbsnr / 10)**2
            sig_scaled = signal * (avg_brightness / signal.mean())
            out = poisson.rvs(sig_scaled) * (signal.mean() / avg_brightness)
    elif noise_model == None:
        return frames
    else:
        raise Exception("Invalid noise model '{}'".format(noise_model))

    return out

def test_sequence(
        num_frames=30, dbsnr=-25, drift=(10, 10), rotation_std=0, jitter_std=0,
        frame_size=(250, 250)
):
    """Build noisy test image sequence

    Returns:
        frames_clean
        frames
    """
    from skimage.data import hubble_deep_field
    scene = hubble_deep_field()[:, :, 0]

    frames_clean = get_frames(
        scene=scene,
        drift=drift,
        rotation_std=rotation_std,
        resolution_ratio=1,
        frame_size=frame_size,
        jitter_std=jitter_std,
        num_frames=num_frames,
        start=(0, 0),
    )

    frames = add_noise(
        frames_clean,
        noise_model='gaussian',
        dbsnr=dbsnr,
    )

    return frames_clean, frames
