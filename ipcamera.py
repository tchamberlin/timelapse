"""Take interval-based images from a webcam"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging
import subprocess

import cv2 as cv

logger = logging.getLogger(__name__)


def make_rtsp_uri(host, port, stream_id, username, password):
    return f"rtsp://{host}:{port}/{stream_id}?username={username}&password={password}"


def do_captures(camera, interval: timedelta, filename_format: str, output=Path(".")):
    next_capture = datetime.now()
    enabled = False
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
                camera.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
                camera.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
                # camera.set(cv.CAP_PROP_AUTOFOCUS, 1)
                logger.debug(
                    f"Set resolution to: {camera.get(cv.CAP_PROP_FRAME_WIDTH)}x{camera.get(cv.CAP_PROP_FRAME_HEIGHT)}"
                )
            # Now attempt to read a frame from the camera
            enabled, frame = camera.read()
            # If it isn't enabled, release it
            if not enabled:
                logger.error("Webcam disabled! Check connection, etc.")
                # camera.release()
            # If it is working, write the frame to file
            else:
                cv.imwrite(str(output_path), frame)
                print(f"Wrote {output_path}")
            logger.debug(f"Next capture at {next_capture}")


def do_opencv(
    host,
    interval,
    port=554,
    output=Path("."),
    filename_timestamp_format="%Y-%m-%d_%H-%M-%S_%f",
    stream_id="stream0",
    username="admin",
    # TODO: Should come from a config file or something
    password="42B69D007ACC57C0131772AABDE24ACF",
    verbose=False,
):
    uri = make_rtsp_uri(host, port, stream_id, username, password)
    next_capture = datetime.now()
    enabled = False
    camera = None
    while True:
        now = datetime.now()
        if now > next_capture:
            logger.debug(f"Off by {now - next_capture}")
            next_capture += interval
            now = datetime.now()
            now_str = now.strftime(filename_timestamp_format)
            filename = f"{now_str}.jpeg"
            output_path = output / filename
            # If camera was disabled iteration, try now
            # to re-initialize it
            if not enabled:
                # Then re-open it
                logger.debug(f"uri: {uri}")
                camera = cv.VideoCapture(uri)
                # camera.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
                # camera.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
                # camera.set(cv.CAP_PROP_AUTOFOCUS, 1)
                logger.debug(
                    f"Set resolution to: {camera.get(cv.CAP_PROP_FRAME_WIDTH)}x{camera.get(cv.CAP_PROP_FRAME_HEIGHT)}"
                )
            # Now attempt to read a frame from the camera
            enabled, frame = camera.read()
            # If it isn't enabled, release it
            if not enabled:
                logger.error("Webcam disabled! Check connection, etc.")
                camera.release()
            # If it is working, write the frame to file
            else:
                cv.imwrite(str(output_path), frame)
                print(f"Wrote {output_path}")
            logger.debug(f"Next capture at {next_capture}")


def do_ffmpeg(
    host,
    fps=None,
    interval=None,
    port=554,
    output=Path("."),
    filename_format=None,
    # filename_format="img%09d.png",
    filename_timestamp_format="%Y-%m-%d_%H-%M-%S.png",
    stream_id="stream0",
    username="admin",
    # TODO: Should come from a config file or something
    password="42B69D007ACC57C0131772AABDE24ACF",
    verbose=False,
):
    if fps is None and interval is None:
        raise ValueError("fps or interval required")

    # If given an interval (e.g. 5 seconds-per-frame), convert to frames-per-second
    if interval:
        fps = 1 / interval.total_seconds()
        logger.debug(
            f"Converted {interval.total_seconds()} seconds-per-frame to {fps} frames-per-second"
        )

    output_args = (
        ["-strftime", "1", str(output / filename_timestamp_format)]
        if filename_timestamp_format
        else [str(output / filename_format)]
    )
    quiet_args = [] if verbose else ["-v", "quiet"]
    ffmpeg_cmd_list = [
        "ffmpeg",
        "-rtsp_transport",
        "tcp",
        "-i",
        make_rtsp_uri(host, port, stream_id, username, password),
        "-f",
        "image2",
        # "-r",
        # f"1/{interval.total_seconds()}",
        "-vf",
        f"fps=fps={fps}",
        *quiet_args,
        *output_args,
    ]
    logger.debug(f"ffmpeg_cmd: {' '.join(ffmpeg_cmd_list)!r}")
    try:
        result = subprocess.check_output(ffmpeg_cmd_list, universal_newlines=True)
    except subprocess.CalledProcessError as error:
        logger.exception("ffmpeg error!")
        result = None
    except FileNotFoundError as error:
        logger.exception("Failed to find ffmpeg in PATH (probably)!")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("host", help="Hostname or IP of the IP camera")
    parser.add_argument("--port", default=554, help="Port to use on the IP camera")
    parser.add_argument(
        "-i",
        "--interval",
        type=lambda x: timedelta(seconds=int(x)),
        help="Interval between captures, in seconds",
    )

    parser.add_argument("--fps", type=int, help="FPS to capture")

    parser.add_argument("--format", default="img%09d.png")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--backend", choices=("ffmpeg", "opencv"), default="ffmpeg")
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

    if args.backend == "ffmpeg":
        logger.debug("Using ffmpeg backend")
        camera_func = do_ffmpeg
    else:
        logger.debug("Using opencv backend")
        camera_func = do_opencv

    camera_func(
        output=args.output,
        interval=args.interval,
        # filename_format=args.format,
        host=args.host,
        port=args.port,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
