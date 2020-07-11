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

        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("COXFaceDBStillExtractor")
    parser.add_argument("input_dir", metavar="DIR")
    parser.add_argument("eye_location_path", metavar="PATH")
    parser.add_argument("output_dir", metavar="DIR")
    parser.add_argument("--factor", "-f", default=2.0, type=float)
    parser.add_argument("--pad-mode", default="edge")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args.input_dir,
         args.eye_location_path,
         args.output_dir,
         args.factor,
         args.pad_mode,
         args.debug)
