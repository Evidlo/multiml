#!/usr/bin/env python3

from multiml.observation import test_sequence
from multiml.multiml import register, shift_and_sum
import matplotlib.pyplot as plt

# get test image sequence
drift = [10, 10]
clean, noisy = test_sequence(dbsnr=-20)

# register and reconstruct
drift_estimate = register(noisy)
reconstruction = shift_and_sum(noisy, drift_estimate, mode='crop')


def imshow_subplot(image, rows, cols, num):
    plt.subplot(rows, cols, num)
    plt.imshow(image)
    plt.axis(False)

plt.subplot(3, 5, 1)
plt.text(0, -10, 'Clean Frames')
imshow_subplot(clean[0], 3, 5, 1)
imshow_subplot(clean[5], 3, 5, 2)
imshow_subplot(clean[10], 3, 5, 3)
imshow_subplot(clean[15], 3, 5, 4)
imshow_subplot(clean[20], 3, 5, 5)

plt.subplot(3, 5, 6)
plt.text(0, -10, 'Noisy Frames')
imshow_subplot(noisy[0], 3, 5, 6)
imshow_subplot(noisy[5], 3, 5, 7)
imshow_subplot(noisy[10], 3, 5, 8)
imshow_subplot(noisy[15], 3, 5, 9)
imshow_subplot(noisy[20], 3, 5, 10)

plt.subplot(3, 5, 13)
plt.text(0, -10, 'Reconstruction')
imshow_subplot(reconstruction, 3, 5, 13)
plt.subplot(3, 5, 12)
plt.text(0, .5, f"Actual drift\n{drift}\nDrift estimate\n{drift_estimate}")
plt.axis(False)

plt.tight_layout()
plt.show()
