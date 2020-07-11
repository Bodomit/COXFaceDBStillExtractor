import os
import argparse
import csv
import glob
import functools

import numpy as np
import skimage.io
import skimage.util
import skimage.draw

from typing import Dict, Tuple, Set
Coord = Tuple[int, int]
EyeLocs = Tuple[Coord, Coord]


def load_eye_locations(eye_location_path: str) -> Dict[str, EyeLocs]:
    eye_locations_for_still: Dict[str, EyeLocs] = {}
    with open(eye_location_path, "rt") as infile:
        reader = csv.reader(infile, delimiter=" ")
        for row in reader:
            path = row[0]
            eye_left = (int(row[2]), int(row[1]))
            eye_right = (int(row[4]), int(row[3]))
            eye_locations_for_still[path] = (eye_left, eye_right)
    return eye_locations_for_still


def load_paths(input_dir: str) -> Set[str]:
    pattern = os.path.join(input_dir, "*.JPG")
    paths = glob.glob(pattern)
    return set(paths)


def debug_show(debug: bool, image: np.ndarray, eye_locs: EyeLocs = None):
    image = image.copy()
    if debug:
        if eye_locs:
            for eye_loc in eye_locs:
                rr, cc = skimage.draw.circle(*eye_loc, 5)
                image[rr, cc, 0] = 255

        skimage.io.imshow(image)
        skimage.io.show()


def pad_axis(crop: Tuple[int, int]) \
        -> Tuple[Tuple[int, int], Tuple[int, int]]:
    pad_before = 0
    pad_after = 0
    if crop[0] < 0:
        pad_before = np.absolute(crop[0])
        crop = (0, crop[1])
    if crop[1] < 0:
        pad_after = np.absolute(crop[1])
        crop = (crop[0], 0)
    return (pad_before, pad_after), crop


def pad(image: np.ndarray,
        crop_height: Tuple[int, int],
        crop_width: Tuple[int, int],
        pad_mode: str) -> np.ndarray:

    pad_height, crop_height = pad_axis(crop_height)
    pad_width, crop_width = pad_axis(crop_width)

    image = skimage.util.pad(image, (pad_height, pad_width, (0, 0)), pad_mode)
    return image, crop_height, crop_width


def crop_axis(old: int, new: int, center: int) -> Tuple[int, int]:
    point_1 = center - new // 2
    point_2 = center + new // 2
    return (point_1, old-point_2)


def crop(image: np.ndarray,
         new_height: int,
         new_width: int,
         center: Coord,
         pad_mode: str) -> np.ndarray:
    old_height, old_width = image.shape[0:2]
    crop_height = crop_axis(old_height, new_height, center[0])
    crop_width = crop_axis(old_width, new_width, center[1])

    image, crop_height, crop_width = pad(image,
                                         crop_height,
                                         crop_width,
                                         pad_mode)
    image = skimage.util.crop(image, (crop_height, crop_width, (0, 0)))

    return image


def eye_centerpoint(left_eye: Coord,
                    right_eye: Coord) -> Coord:
    center_height = (left_eye[0] + right_eye[0]) // 2
    center_width = (left_eye[1] + right_eye[1]) // 2
    return (center_height, center_width)


def main(input_dir: str,
         eye_location_path: str,
         output_dir: str,
         factor: float,
         pad_mode: str,
         debug: bool):

    eye_locations = load_eye_locations(eye_location_path)
    image_paths = load_paths(input_dir)
    assert len(eye_locations) == len(image_paths)

    show = functools.partial(debug_show, debug)

    for image_path in image_paths:
        image = skimage.io.imread(image_path)
        eye_locs = eye_locations[os.path.basename(image_path)]
        show(image, eye_locs)

        left_eye, right_eye = eye_locs
        eye_dist = right_eye[1] - left_eye[1]
        center = eye_centerpoint(left_eye, right_eye)

        height = width = int(eye_dist * factor)
        image = crop(image, height, width, center, pad_mode)

        show(image)

        rel_path = os.path.relpath(image_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        skimage.io.imsave(out_path, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("COXFaceDBStillExtractor")
    parser.add_argument("input_dir", metavar="DIR")
    parser.add_argument("eye_location_path", metavar="PATH")
    parser.add_argument("output_dir", metavar="DIR")
    parser.add_argument("--factor", "-f", default=5.0, type=float)
    parser.add_argument("--pad-mode", default="edge")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args.input_dir,
         args.eye_location_path,
         args.output_dir,
         args.factor,
         args.pad_mode,
         args.debug)
