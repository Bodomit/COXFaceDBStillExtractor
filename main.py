import os
import argparse


def main(input_dir: str,
         eye_location_path: str,
         output_dir: str,
         factor: float,
         pad_mode: str):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("COXFaceDBStillExtractor")
    parser.add_argument("input_dir", metavar="DIR")
    parser.add_argument("eye_location_path", metavar="PATH")
    parser.add_argument("output_dir", metavar="DIR")
    parser.add_argument("--factor", "-f", default=2.0, type=float)
    parser.add_argument("--pad-mode", default="edge")

    args = parser.parse_args()
    main(args.input_dir,
         args.eye_location_path,
         args.output_dir,
         args.factor,
         args.pad_mode)
