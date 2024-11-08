#!/usr/bin/python

# Copyright (c) 2015 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This is the code behind the Switching Eds blog post:

    http://matthewearl.github.io/2015/07/28/switching-eds-with-python/

See the above for an explanation of the code below.

To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV. You'll also need to obtain the trained model from
sourceforge:

    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

Unzip with `bunzip2` and change `PREDICTOR_PATH` to refer to this file. The
script is run like so:

    ./faceswap.py <head image> <face image>

If successful, a file `output.jpg` will be produced with the facial features
from `<head image>` replaced with the facial features from `<face image>`.

"""

#!/usr/bin/python

import cv2
import dlib
import numpy as np
import sys
import logging

# Constants
PREDICTOR_PATH = "/home/matt/dlib-18.16/shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1  # You can adjust this for faster processing if necessary
FEATHER_AMOUNT = 11
COLOUR_CORRECT_BLUR_FRAC = 0.6

# Facial landmark groupings
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used for image alignment
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points for overlay in face swap
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

logging.basicConfig(filename='faceswap.log', level=logging.INFO)


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def detect_faces(im):
    """Detect faces in the image and return their bounding boxes."""
    rects = detector(im, 1)
    if len(rects) == 0:
        raise NoFaces("No faces detected.")
    if len(rects) > 1:
        logging.warning(f"Multiple faces detected. Selecting the first face.")
    return rects[0]  # Return the first face (if multiple)


def get_landmarks(im, rect):
    """Get the facial landmarks from the detected face."""
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])


def annotate_landmarks(im, landmarks):
    """Annotate the image with facial landmarks (for debugging purposes)."""
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4, color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def draw_convex_hull(im, points, color):
    """Draw a convex hull over the given points on the image."""
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    """Generate a face mask from the landmarks."""
    im_mask = np.zeros(im.shape[:2], dtype=np.float64)
    for group in OVERLAY_POINTS:
        draw_convex_hull(im_mask, landmarks[group], color=1)
    im_mask = np.array([im_mask, im_mask, im_mask]).transpose((1, 2, 0))
    im_mask = (cv2.GaussianBlur(im_mask, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    return cv2.GaussianBlur(im_mask, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)


def transformation_from_points(points1, points2):
    """Calculate the affine transformation matrix that maps points1 to points2."""
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])


def correct_colours(im1, im2, landmarks1):
    """Correct the color differences between two images."""
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) - np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


def seamless_clone(im1, im2, mask, center):
    """Blend two images seamlessly using OpenCV's seamlessClone."""
    return cv2.seamlessClone(im2, im1, mask.astype(np.uint8), center, cv2.NORMAL_CLONE)


def read_im_and_landmarks(fname):
    """Read an image and return it with its landmarks."""
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR))
    rect = detect_faces(im)
    landmarks = get_landmarks(im, rect)
    return im, landmarks


def warp_im(im, M, dshape):
    """Warp an image using an affine transformation."""
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im, M[:2], (dshape[1], dshape[0]), dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
    return output_im


def main():
    try:
        # Read input images
        im1, landmarks1 = read_im_and_landmarks(sys.argv[1])
        im2, landmarks2 = read_im_and_landmarks(sys.argv[2])

        # Calculate transformation matrix
        M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])

        # Generate face masks
        mask = get_face_mask(im2, landmarks2)
        warped_mask = warp_im(mask, M, im1.shape)
        combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)

        # Warp the second image onto the first image
        warped_im2 = warp_im(im2, M, im1.shape)
        warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

        # Perform seamless cloning for better blending
        center = (im1.shape[1] // 2, im1.shape[0] // 2)  # Center of the face
        output_im = seamless_clone(im1, warped_corrected_im2, combined_mask, center)

        # Save output image
        cv2.imwrite('output.jpg', output_im)

    except NoFaces as e:
        print(f"Error: {e}")
        logging.error(f"Error: {e}")

    except TooManyFaces as e:
        print(f"Error: {e}")
        logging.error(f"Error: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")
        logging.exception("Unexpected error during face swap.")


if __name__ == "__main__":
    main()
