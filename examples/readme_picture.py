#!/usr/bin/env python3

from multiml.observation import test_sequence
from multiml.multiml import register, shift_and_sum
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# get test image sequence
drift = [10, 10]
dbsnr=-20
clean, noisy = test_sequence(dbsnr=dbsnr)

# register and reconstruct
drift_estimate = register(noisy)
reconstruction = shift_and_sum(noisy, drift_estimate, mode='crop')

# plot
fig, subplots = plt.subplots(2, 2)
clean_plot = subplots[0][0]
clean_plot.set_title("Clean Frames")
noisy_plot = subplots[0][1]
noisy_plot.set_title(f"Noisy Frames ({dbsnr} dB)")
result_plot = subplots[1][0]
result_plot.text(.5, .5, f"Actual drift\n{drift}\nEstimated drift\n{drift_estimate}")
result_plot.axis(False)
recon_plot = subplots[1][1]
recon_plot.set_title("Registered + Coadded")
recon_plot.imshow(reconstruction)

plt.tight_layout()

def update(i):
    clean_plot.imshow(clean[i])
    noisy_plot.imshow(noisy[i])


# run frame sequence forward and back
fps = 15
anim = animation.FuncAnimation(
    fig, update,
    frames=[*range(len(clean)), *range(len(clean)-1, -1, -1)],
    interval=1000//fps,
)

def saveit():
    writergif = animation.PillowWriter(fps=fps)
    anim.save('../readme_picture.gif', writergif)
