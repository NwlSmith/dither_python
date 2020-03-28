# Python code to reading an image using OpenCV
import cv2
import numpy as np
import random
import math
import noise
import sys
from copy import copy


# Create a scaled-down version of the image to better show dithering effects
def scale_down_img():
    sdsq = scale_down * scale_down
    for y in range(h - scale_down + 1):
        if y % scale_down != 0:
            continue
        # for every row
        for x in range(w - scale_down + 1):
            if x % scale_down != 0:
                continue

            total = 0
            for y2 in range(scale_down):
                for x2 in range(scale_down):
                    total += img[y + y2, x + x2]
            img2[int(y / scale_down), int(x / scale_down)] = int(total / sdsq)


# The upper modulate function for dither_mod
def dither_mod_upper(y, x, mod):
    if (y + x) % mod != 0:
        img2[y, x] = 255
    else:
        img2[y, x] = 0


# The lower modulate function for dither_mod
def dither_mod_lower(y, x, mod):
    if (y + x) % mod == 0:
        img2[y, x] = 255
    else:
        img2[y, x] = 0


# At each color threshold, color the image according to a modulate function.
# Colors the image through white/black lines of varying thickness.
# used to create selfie1.jpg and selfie3.jpg, although a modified (x * y) instead of (x + y) was used in selfie2.jpg
def dither_mod(y, x):
    if img2[y, x] >= 225:
        img2[y, x] = 255
    elif img2[y, x] >= 200:
        dither_mod_upper(y, x, 6)
    elif img2[y, x] >= 175:
        dither_mod_upper(y, x, 5)
    elif img2[y, x] >= 150:
        dither_mod_upper(y, x, 4)
    elif img2[y, x] >= 125:
        dither_mod_upper(y, x, 3)
    elif img2[y, x] >= 100:
        dither_mod_lower(y, x, 2)
    elif img2[y, x] >= 100:
        dither_mod_lower(y, x, 3)
    elif img2[y, x] >= 100:
        dither_mod_lower(y, x, 4)
    elif img2[y, x] >= 75:
        dither_mod_lower(y, x, 5)
    elif img2[y, x] >= 50:
        dither_mod_lower(y, x, 6)
    elif img2[y, x] >= 25:
        dither_mod_lower(y, x, 7)


# Random Dither algorithm.
# Randomly choose a number and if the number is greater than the color at that pixel, set it to black
# Resembles a noise function. Used to create selfie4.jpg
def dither_rand(y, x):
    if random.randrange(255) <= img2[y, x]:
        img2[y, x] = 255
    else:
        img2[y, x] = 0


# Dither image according to sin waves.
# Like dither_rand, this algorithm is similar to a noise function, but it is non-random and consistent. Image is colored
# by lines of varying thickness. Used to create selfies 5, 6 ((y * x), not (y + x)), 7, 8, and 9.
def dither_sin(y, x):
    if math.sin(30*(y + x)) * 255 <= img2[y, x]:
        img2[y, x] = 255
    else:
        img2[y, x] = 0


# Floyd Steinberg algorithm
# Each the decision of whether or not to change a pixel influences the adjacent pixels (down-left, down, down-right, and
# right). Used in selfie 11. Results in odd artifacts occasionally
def dither_floyd_steinberg(y, x):
    old_pix = copy(img2[y, x][0])
    new_pix = round(old_pix / 255) * 255
    img2[y, x] = new_pix
    err = old_pix - new_pix
    if x < w2 - 1:
        img2[y, x + 1] = img2[y, x + 1] + err * 7 / 16
    if y < h2 - 1 and x > 0:
        img2[y + 1, x - 1] = img2[y + 1, x - 1] + err * 3 / 16
    if y < h2 - 1:
        img2[y + 1, x] = img2[y + 1, x] + err * 5 / 16
    if y < h2 - 1 and x < w2 - 1:
        img2[y + 1, x + 1] = img2[y + 1, x + 1] + err * 1 / 16


# Definition of Bayer pattern used in dither_bayer().
def bayer_patterns():
    global bayer
    bayer = [
        [ # 0
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ],
        [  # 1
            [255, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ],
        [  # 2
            [255, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 255, 0],
            [0, 0, 0, 0]
        ],
        [  # 3
            [255, 0, 0, 0],
            [0, 0, 0, 0],
            [255, 0, 255, 0],
            [0, 0, 0, 0]
        ],
        [  # 4
            [255, 0, 255, 0],
            [0, 0, 0, 0],
            [255, 0, 255, 0],
            [0, 0, 0, 0]
        ],
        [  # 5
            [255, 0, 255, 0],
            [0, 255, 0, 0],
            [255, 0, 255, 0],
            [0, 0, 0, 0]
        ],
        [  # 6
            [255, 0, 255, 0],
            [0, 255, 0, 0],
            [255, 0, 255, 0],
            [0, 0, 0, 255]
        ],
        [  # 7
            [255, 0, 255, 0],
            [0, 255, 0, 0],
            [255, 0, 255, 0],
            [0, 255, 0, 255]
        ],
        [  # 8 - HALFWAY
            [255, 0, 255, 0],
            [0, 255, 0, 255],
            [255, 0, 255, 0],
            [0, 255, 0, 255]
        ],
        [  # 9
            [255, 0, 255, 0],
            [255, 255, 0, 255],
            [255, 0, 255, 0],
            [0, 255, 0, 255]
        ],
        [  # 10
            [255, 0, 255, 0],
            [255, 255, 0, 255],
            [255, 0, 255, 0],
            [0, 255, 255, 255]
        ],
        [  # 11
            [255, 0, 255, 0],
            [255, 255, 0, 255],
            [255, 0, 255, 0],
            [255, 255, 255, 255]
        ],
        [  # 12
            [255, 0, 255, 0],
            [255, 255, 255, 255],
            [255, 0, 255, 0],
            [255, 255, 255, 255]
        ],
        [  # 13
            [255, 255, 255, 0],
            [255, 255, 255, 255],
            [255, 0, 255, 0],
            [255, 255, 255, 255]
        ],
        [  # 14
            [255, 255, 255, 0],
            [255, 255, 255, 255],
            [255, 0, 255, 255],
            [255, 255, 255, 255]
        ],
        [  # 15
            [255, 255, 255, 0],
            [255, 255, 255, 255],
            [255, 255, 255, 255],
            [255, 255, 255, 255]
        ],
        [  # 16
            [255, 255, 255, 255],
            [255, 255, 255, 255],
            [255, 255, 255, 255],
            [255, 255, 255, 255]
        ]
    ]


# Bayer algorithm.
# Each pixel references a pixel on the bayer pattern array according to its placement and color.
# Used to create selfie 12 (an error made the darkest black appear white), 13 (an error prevented the whitest white from
# showing), 14, and 15 (which included random variances).
def dither_bayer(y, x):
    #if bayer_rand and 25 < img2[y, x][0] < 230:
        #img2[y, x][0] += random.randrange(-15, 15)
    c = int(img2[y, x][0] * 16 / 255)
    img2[y, x] = bayer[c][y % 4][x % 4]


# Perlin Noise algorithm.
def dither_noise(y, x):
    c = img2[y, x][0] / 255
    n = noise.pnoise1(c * 100)
    n = (n + 1) / 2
    if n > c:
        img2[y, x] = [0]
    else:
        img2[y, x] = [255]


# Dither this pixel according to the indicated algorithm.
def dither(alg):
    #
    for y in range(int(h / scale_down)):
        # for every row
        for x in range(int(w / scale_down)):
            if alg is 1:
                dither_mod(y, x)
            elif alg is 2:
                dither_rand(y, x)
            elif alg is 3:
                dither_sin(y, x)
            elif alg is 4:
                dither_floyd_steinberg(y, x)
            elif alg is 5:
                dither_bayer(y, x)
            elif alg is 6:
                dither_noise(y, x)
            else:
                print("Incorrect dither algorithm number input. Please choose a number between 1 and 5.")


if len(sys.argv) < 4:
    print("Must provide 3 arguments: image path, dither algorithm, scale_down")
    sys.exit()

# Define bayer patterns
bayer_patterns()

# Define which dither algorithm will be used
dit_alg = 6
if sys.argv[2] is not None:
    dit_alg = int(sys.argv[2])
bayer_rand = False

# You can give path to the image as first argument
if sys.argv[1] is None:
    print("Please provide an image path")
    sys.exit()
img = cv2.imread(sys.argv[1], 0)

# retrieve dimensions of original img
h, w = img.shape[:2]

scale_down = 4
scale_up = scale_down
if sys.argv[3] is not None:
    scale_down = int(sys.argv[3])
    scale_up = scale_down

h2 = int(h / scale_down)
w2 = int(w / scale_down)

print(h, w)

# create new image as scaled-down reference
img2 = np.zeros((int(h / scale_down), int(w / scale_down), 1), np.uint8)
scale_down_img()

dither(dit_alg)

# create new final image
img3 = np.zeros((h, w, 1), np.uint8)
# Blow that back up as a test
for y3 in range(h-scale_down):
    for x3 in range(w-scale_down):
        img3[y3, x3] = img2[int(y3/scale_down), int(x3/scale_down)]

# will show the image in a window
cv2.imshow('image', img)
cv2.imshow('image2', img2)
cv2.imshow('image3', img3)

k = cv2.waitKey(0) & 0xFF

# wait for ESC key to exit
if k == 27:
    cv2.destroyAllWindows()

# wait for 's' key to save and exit
elif k == ord('s'):
    cv2.imwrite('messigray.png', img3)
    cv2.destroyAllWindows()