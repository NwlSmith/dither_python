# dither_python
- Nate Smith
- 12/30/2019
Experiment recreating dithering algorithms in python, cv2 and numpy

To protect my privacy, the notes "selfies" in the comments of ditherv1.py are not included in the public repository.

To run, run the command line prompt

> python ditherv1.py "[image-to-dither]" [Dither-algorithm] [image-downsizing]

[image-to-dither] must be in quotes

Dither-algorithms:

1 - At each color threshold, this algorithm colors the image according to a modulate function. It colors the image through white/black lines of varying thickness.

2 - Randomly chooses a number and if the number is greater than the color at that pixel, sets it to black.

3 - Dither image according to sin waves. Like 1, this algorithm is similar to a noise function, but it is non-random and consistent. Image is colored by lines of varying thickness.

4 - Floyd Steinberg algorithm. The decision of whether or not to change a pixel influences the adjacent pixels (down-left, down, down-right, and right).

5 - Bayer algorithm. Each pixel references a pixel on the bayer pattern array according to its placement and color.

6 - Perlin Noise algorithm.