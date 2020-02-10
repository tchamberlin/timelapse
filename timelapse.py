"""Make timelapse from images"""

from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from pprint import pformat
import argparse
import contextlib
import logging
import os
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile

from dateutil import parser as dp
import ffmpeg
import gevent
import gevent.monkey
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 2020-01-21_20-55-49_120
THE_FORMAT = "%Y-%m-%d_%H-%M-%S_%f"

FFMPEG_PROGRESS_REGEX = re.compile(r".+time=([\d:\.]+).+")


# TODO: Get rid of gevent and use threads?
gevent.monkey.patch_all(thread=False)


@contextlib.contextmanager
def _tmpdir_scope():
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)


def _do_watch_progress(filename, sock, handler):
    """Function to run in a separate gevent greenlet to read progress
    events from a unix-domain socket."""
    connection, client_address = sock.accept()
    data = b""
    try:
        while True:
            more_data = connection.recv(16)
            if not more_data:
                break
            data += more_data
            lines = data.split(b"\n")
            for line in lines[:-1]:
                line = line.decode()
                parts = line.split("=")
                key = parts[0] if len(parts) > 0 else None
                value = parts[1] if len(parts) > 1 else None
                handler(key, value)
            data = lines[-1]
    finally:
        connection.close()


@contextlib.contextmanager
def _watch_progress(handler):
    """Context manager for creating a unix-domain socket and listen for
    ffmpeg progress events.

    The socket filename is yielded from the context manager and the
    socket is closed when the context manager is exited.

    Args:
        handler: a function to be called when progress events are
            received; receives a ``key`` argument and ``value``
            argument. (The example ``show_progress`` below uses tqdm)

    Yields:
        socket_filename: the name of the socket file.
    """
    with _tmpdir_scope() as tmpdir:
        socket_filename = os.path.join(tmpdir, "sock")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        with contextlib.closing(sock):
            sock.bind(socket_filename)
            sock.listen(1)
            child = gevent.spawn(_do_watch_progress, socket_filename, sock, handler)
            try:
                yield socket_filename
            except:
                gevent.kill(child)
                raise


@contextlib.contextmanager
def show_progress(total_duration):
    """Create a unix-domain socket to watch progress and render tqdm
    progress bar."""
    with tqdm(total=round(total_duration, 2)) as bar:

        def handler(key, value):
            if key == "out_time_ms":
                time = round(float(value) / 1000000., 2)
                if time < 0:
                    time = 0
                bar.update(time - bar.n)
            elif key == "progress" and value == "end":
                bar.update(bar.total - bar.n)

        with _watch_progress(handler) as socket_filename:
            yield socket_filename


def gen_description(stats, input_fps, playback_fps):
    description_lines = []

    # description_lines.append(f"Total gaps remaining: {stats['total_gaps']}")
    # description_lines.append(
    #     f"Num images filtered out: {stats['num_filtered']}/{stats['total_processed']} "
    #     f"({stats['percent_filtered']:.2%})"
    # )
    if "total_size_to_process" in stats:
        description_lines.append(
            f"{stats['total_size_to_process'] / (1 << 30):.2f} "
            "GB of images were processed for this timelapse"
        )

    speedup = stats["speedup"]
    description_lines.append(
        f"{stats['num_to_process']} timelapse images were taken at {stats['avg_delta']} intervals over "
        f"{stats['total_length_to_process']}. "
        f"The resultant timelapse is {stats['timelapse_playback_time']} "
        f"({speedup:.2f}x real-time)."
    )

    intervals = (
        timedelta(minutes=1),
        timedelta(hours=1),
        timedelta(days=1),
        timedelta(weeks=1),
    )

    description_lines.append(f"For context, a {speedup:.2f}x speedup means that: ")
    for interval in intervals:
        description_lines.append(
            f"  {format_td_verbose(interval)} real time "
            f"== {format_td_verbose(interval / speedup)} playback ({interval / speedup})"
        )

    return "\n".join(description_lines)


def format_td_verbose(td):
    seconds = td.total_seconds()

    weeks = seconds // (7 * 60 * 60 * 24)
    seconds = seconds - weeks * (7 * 60 * 60 * 24)

    days = seconds // (60 * 60 * 24)
    seconds = seconds - (days * (60 * 60 * 24))
    hours = seconds // 3600
    seconds = seconds - (hours * 3600)
    minutes = seconds // 60
    seconds = seconds - (minutes * 60)

    sections = []
    if weeks:
        sections.append(f"{weeks:.0f} week{'s' if weeks > 1 else ''}")
    if days:
        sections.append(f"{days:.0f} day{'s' if days > 1 else ''}")
    if hours:
        sections.append(f"{hours:.0f} hour{'s' if hours > 1 else ''}")
    if minutes:
        sections.append(f"{minutes:.0f} minute{'s' if minutes > 1 else ''}")
    if seconds:
        sections.append(f"{seconds:.2f} second{'s' if seconds > 1 else ''}")

    return ", ".join(sections)


def parse_image_path(path: Path, the_format=None):
    if the_format:
        return datetime.strptime(".".join(path.name.split(".")[:-1]), the_format)
    else:
        return datetime.fromtimestamp(path.stat().st_mtime)


def get_files_to_process(
    input_glob,
    expected=5,
    margin=1,
    start=None,
    end=None,
    extended_stats=False,
    time_format=THE_FORMAT,
):
    input_path = os.path.expanduser(input_glob)
    paths = glob(input_path)
    margin_td = timedelta(seconds=margin)
    expected_td = timedelta(seconds=expected)
    paths = sorted([Path(path) for path in paths])
    paths_to_process = []
    total_gaps = timedelta(0)
    weird_length = timedelta(0)

    start_filter_str = f"after {start}"
    end_filter_str = f"before {end}"
    if start and end:
        filter_str = f"{start_filter_str} and {end_filter_str}"
    elif start:
        filter_str = start_filter_str
    elif end:
        filter_str = end_filter_str
    else:
        filter_str = None
    num_filtered = 0
    stats = {
        "total_gaps": 0,
        "total_processed": len(paths),
        "start_filter": start,
        "end_filter": end,
    }
    deltas = []
    if start or end:
        logger.info(f"Keeping only images taken {filter_str}")
    for path, next_path in zip(paths, [*paths[1:], None]):
        if time_format:
            path_dt = parse_image_path(path, time_format)
        else:
            path_dt = datetime.fromtimestamp(path.stat().st_mtime)
        if (start and path_dt < start) or (end and path_dt > end):
            num_filtered += 1
        else:
            paths_to_process.append(path)
            if next_path:
                if time_format:
                    next_path_dt = parse_image_path(next_path, time_format)
                else:
                    next_path_dt = datetime.fromtimestamp(path.stat().st_mtime)

                delta = next_path_dt - path_dt
                if delta - expected_td > margin_td:
                    logger.warning(
                        f"Gap of >{margin_td} between {Path(path).name} and {Path(next_path).name}: {delta}"
                    )
                    total_gaps += delta
                else:
                    weird_length += delta
                deltas.append(delta)

    logger.debug(
        f"{num_filtered} images were filtered out due to "
        f"being taken outside bounds: {filter_str}"
    )

    stats["total_length_with_gaps"] = parse_image_path(
        paths[-1], time_format
    ) - parse_image_path(paths[0], time_format)
    stats["total_length_to_process"] = parse_image_path(
        paths_to_process[-1], time_format
    ) - parse_image_path(paths_to_process[0], time_format)
    stats["total_length_weird"] = weird_length
    stats["total_gaps"] = total_gaps
    stats["num_filtered"] = len(paths) - len(paths_to_process)
    stats["num_to_process"] = len(paths_to_process)
    stats["percent_filtered"] = num_filtered / len(paths) if num_filtered else 0
    stats["min_delta"] = min(deltas)
    stats["max_delta"] = max(deltas)

    logger.debug("Calculating size to process...")
    if extended_stats:
        stats["total_size_to_process"] = sum(
            path.stat().st_size for path in paths_to_process
        )

    stats["avg_delta"] = stats["total_length_to_process"] / stats["num_to_process"]
    off_by = stats["avg_delta"] - expected_td
    if off_by > margin_td:
        logger.warning(
            f"WARNING: the average image interval of {stats['avg_delta']} is greater than "
            f"the expected interval of {expected_td} by {off_by}!"
        )
    logger.debug("...done")
    return paths_to_process, stats


def make_symlinks(paths, padding=9):
    logger.debug(f"Making symlinks...")
    tempdir = tempfile.mkdtemp()
    suffixes = set()
    for i, path in enumerate(paths):
        path = Path(path)
        suffixes.add(path.suffix)
        os.symlink(path, Path(tempdir) / f"{i:0{padding}d}{path.suffix}")
    logger.debug(f"...done")

    if len(suffixes) > 1:
        raise ValueError(f"Multiple file suffixes found in input paths: {suffixes}")
    return Path(tempdir), padding, suffixes.pop()


def timelapse(
    files_to_process,
    expected_playback_time,
    input_fps=30,
    output_fps=30,
    output_path="./out.mp4",
):
    # ffmpeg is BAD at handling multiple input files. To get around this, we first create
    # a bunch of symlinks that are named in a way that ffmpeg understands
    tempdir, padding, suffix = make_symlinks(files_to_process)
    logger.debug(f"Made tmpdir {tempdir}")
    with show_progress(expected_playback_time.total_seconds()) as socket_filename:
        try:
            stdout, stderr = (
                ffmpeg.input(
                    str(os.path.join(tempdir, f"%0{padding}d{suffix}")),
                    framerate=input_fps,
                )
                .crop(width=1272, height=716, x=297, y=0)
                .filter("eq", saturation=1.8, contrast=1.03, brightness=0.02, gamma=0.9)
                .output(
                    str(output_path), r=output_fps, vcodec="libx265", pix_fmt="yuv420p"
                )
                .global_args("-progress", "unix://{}".format(socket_filename))
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            logger.exception(e.stderr.decode("utf-8"), file=sys.stderr)
            sys.exit(1)

    logger.debug(f"Removing temp dir {tempdir}")
    shutil.rmtree(tempdir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default="./timelapse.mp4")

    target_group = parser.add_mutually_exclusive_group(required=False)
    target_group.add_argument("--speedup", type=int)
    target_group.add_argument(
        "--input-fps",
        type=int,
        default=120,
        help="Input framerate of the timelapse. 120 input fps on a images taken every minute "
        "will yield a video that runs at 120 minutes/second (7200x speedup)",
    )

    parser.add_argument(
        "--playback-fps",
        type=int,
        default=30,
        help="The framerate of the output video. Has no effect on playback speed",
    )
    parser.add_argument(
        "--start",
        type=dp.parse,
        help="If given, don't process any images taken before this date",
    )
    parser.add_argument(
        "--end",
        type=dp.parse,
        help="If given, don't process any images taken after this date",
    )
    parser.add_argument("--time-format", help="Format string sent to strptime")
    parser.add_argument("--extended-stats", action="store_true")

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase verbosity. This only affects stderr/logging output.",
    )
    parser.add_argument(
        "-D", "--dry-run", action="store_true", help="Don't make any changes"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        init_logging(logging.DEBUG)
    else:
        init_logging(logging.INFO)
    files_to_process, stats = get_files_to_process(
        args.input,
        start=args.start,
        end=args.end,
        extended_stats=args.extended_stats,
        time_format=args.time_format,
    )

    logger.debug(pformat(stats))
    if args.speedup:
        input_fps = args.speedup / stats["avg_delta"].total_seconds()
    else:
        input_fps = args.input_fps

    timelapse_playback_time = timedelta(seconds=stats["num_to_process"] / input_fps)
    speedup = (
        stats["total_length_to_process"].total_seconds()
        / timelapse_playback_time.total_seconds()
    )
    stats["timelapse_playback_time"] = timelapse_playback_time
    stats["speedup"] = speedup

    print(gen_description(stats, input_fps, args.playback_fps))
    if not args.dry_run:
        timelapse(
            files_to_process,
            input_fps=input_fps,
            output_fps=args.playback_fps,
            output_path=args.output,
            expected_playback_time=stats["timelapse_playback_time"],
        )
    else:
        logger.debug("Skipping timelapse processing due to presence of --dry-run")


def init_logging(level):
    """Initialize logging"""
    logging.getLogger().setLevel(level)
    _logger = logging.getLogger(__name__)
    # console_handler = TqdmLoggingHandler()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(console_handler)
    _logger.setLevel(level)


if __name__ == "__main__":
    main()
