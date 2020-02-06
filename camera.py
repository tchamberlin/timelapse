"""Take interval-based images from a webcam"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging

import cv2 as cv

logger = logging.getLogger(__name__)


def camera(interval: timedelta, filename_format: str, camera_index=0, output=Path(".")):
    next_capture = datetime.now()
    enabled = False
    vc = None
    if camera_index is None:
        raise ValueError("Must give camera_index")
    while True:
        now = datetime.now()
        if now > next_capture:
            logger.debug(f"Off by {now - next_capture}")
            next_capture += interval
            now = datetime.now()
            now_str = now.strftime(filename_format)
            filename = f"{now_str}.jpeg"
            output_path = output / filename
            # If camera was disabled iteration, try now
            # to re-initialize it
            if not enabled:
                # Then re-open it
                vc = cv.VideoCapture(camera_index)
                vc.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
                vc.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
                # vc.set(cv.CAP_PROP_AUTOFOCUS, 1)
                logger.debug(
                    f"Set resolution to: {vc.get(cv.CAP_PROP_FRAME_WIDTH)}x{vc.get(cv.CAP_PROP_FRAME_HEIGHT)}"
                )
            # Now attempt to read a frame from the camera
            enabled, frame = vc.read()
            # If it isn't enabled, release it
            if not enabled:
                logger.error("Webcam disabled! Check connection, etc.")
                vc.release()
            # If it is working, write the frame to file
            else:
                cv.imwrite(str(output_path), frame)
                print(f"Wrote {output_path}")
            logger.debug(f"Next capture at {next_capture}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "interval",
        type=lambda x: timedelta(seconds=int(x)),
        help="Interval between captures, in seconds",
    )
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument(
        "-d", "--device-index", type=int, default=0, help="Index of webcam video device"
    )
    parser.add_argument("--format", default="%Y-%m-%d_%H-%M-%S.%f")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def init_logging(level):
    """Initialize logging"""
    logging.getLogger().setLevel(level)
    _logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(console_handler)
    _logger.setLevel(level)


def main():
    args = parse_args()
    if args.verbose:
        init_logging(logging.DEBUG)
    else:
        init_logging(logging.INFO)
    camera(
        args.interval,
        filename_format=args.format,
        camera_index=args.device_index,
        output=args.output,
    )


if __name__ == "__main__":
    main()
