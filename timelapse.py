"""Make timelapse from images"""

from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from pprint import pformat
import re
import sys
import shlex
import pytz

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from dateutil import parser as dp
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 2020-01-21_20-55-49_120
THE_FORMAT = "%Y-%m-%d_%H-%M-%S_%f"

FFMPEG_PROGRESS_REGEX = re.compile(r".+time=([\d:\.]+).+")


def gen_description(stats, input_fps, playback_fps):
    description_lines = []

    description_lines.append(f"Total gaps remaining: {stats['total_gaps']}")
    description_lines.append(
        f"Num images filtered out: {stats['num_filtered']}/{stats['total_processed']} "
        f"({stats['percent_filtered']:.2%})"
    )
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


def parse_image_path(path: Path, the_format=None, tz=None):
    if the_format:
        dt = datetime.strptime(".".join(path.name.split(".")[:-1]), the_format)
    else:
        dt = datetime.fromtimestamp(path.stat().st_mtime)

    return dt


def get_files_to_process(
    input_glob,
    expected=60,
    margin=1,
    start=None,
    end=None,
    extended_stats=False,
    timestamp_source="filename",
    time_format=THE_FORMAT,
    no_progress=False,
    tz=pytz.timezone("America/New_York"),
):
    input_path = os.path.expanduser(input_glob)
    paths = glob(input_path)
    margin_td = timedelta(seconds=margin)
    expected_td = timedelta(seconds=expected)
    paths = sorted([Path(path) for path in paths])
    paths_to_process = []
    total_gaps = timedelta(0)

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
    if timestamp_source == "filename":
        if time_format is None:
            raise ValueError("If timestamp_source is filename, must give time_format!")
        logger.debug(
            f"Parsing image file timestamps using timestamp format {time_format}"
        )
    else:
        logger.debug(f"Parsing image file timestamps using file modification times")

    if start or end:
        logger.info(f"Keeping only images taken {filter_str}")
    for path, next_path in tqdm(
        zip(paths, [*paths[1:], None]), unit="file", disable=no_progress
    ):
        if timestamp_source == "filename":
            path_dt = parse_image_path(path, time_format)
        else:
            path_dt = datetime.fromtimestamp(path.stat().st_mtime)

        if (start and path_dt < start) or (end and path_dt > end):
            # logger.debug(f"Filtered {path}")
            num_filtered += 1
        else:
            paths_to_process.append(path)
            if next_path:
                if timestamp_source == "filename":
                    next_path_dt = parse_image_path(next_path, time_format)
                else:
                    next_path_dt = datetime.fromtimestamp(path.stat().st_mtime)

                delta = next_path_dt - path_dt
                if delta - expected_td > margin_td:
                    logger.warning(
                        f"Gap of >{margin_td} between {Path(path).name} and {Path(next_path).name}: {delta}"
                    )
                    total_gaps += delta
                deltas.append(delta)

    logger.debug(f"{num_filtered} images were filtered out due to given time filter(s)")

    stats["total_length_with_gaps"] = parse_image_path(
        paths[-1], time_format
    ) - parse_image_path(paths[0], time_format)
    stats["total_length_to_process"] = parse_image_path(
        paths_to_process[-1], time_format
    ) - parse_image_path(paths_to_process[0], time_format)
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


def edit_image(in_path, out_path, tz=None):
    img = Image.open(in_path)
    img = img.crop((297, 0, 1272 + 297, 716))
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("LiberationSans-Regular.ttf", 24)
    # font = ImageFont.load_default()
    # draw.text((x, y),"Sample Text",(r,g,b))
    timestamp = parse_image_path(in_path, the_format=THE_FORMAT)
    # if timestamp.tzinfo:
    #     timestr = timestamp.strftime("%Y-%m-%d %H:%M %z")
    # else:
    #     timestr = timestamp.strftime("%Y-%m-%d %H:%M")
    timestr = timestamp.strftime("%Y-%m-%d %H:%M")
    if tz:
        timestr = f"{timestr} {tz}"
    # logger.debug(f"Copy from {in_path} to {out_path}")
    draw.rectangle((0, 690, 200, 716), fill=(0, 0, 0))
    draw.text((5, 690), timestr, (255, 255, 255), font=font)

    img.save(out_path)


def edit_images(
    paths,
    output_dir,
    padding=9,
    skip_exists=True,
    no_progress=False,
    tz=None,
    dry_run=False,
):
    skipped_count = 0
    if dry_run:
        logger.info("Skipping file copying/editing due to dry_run=True")
    else:
        logger.info(f"Copying/modifying input images to {output_dir}...")
    skipped = []
    # Need miniters to avoid 10 sec lag after a bunch of files are skipped
    progress = tqdm(paths, unit="file", disable=no_progress, miniters=1)
    for input_path in progress:
        output_path = output_dir / input_path.name
        if not (skip_exists and output_path.exists()):
            # If we are doing a copy, AND we have previously skipped at least 1 file, print a summary
            # of how many were skipped total
            if skipped_count > 0:
                logger.debug(f"Skipped {skipped_count} images (already exist!)")
            if not dry_run:
                edit_image(input_path, output_path, tz=tz)
            skipped_count = 0
        else:
            skipped.append(output_path)
            skipped_count += 1
        progress.set_description(output_path.name)

    return skipped


def make_symlinks(paths, padding=9, dry_run=False):

    if dry_run:
        logger.debug(f"Skipping symlink creation due to dry_run=True")
    else:
        logger.debug(f"Making symlinks...")
    tempdir = tempfile.mkdtemp()
    suffixes = set()
    for i, path in enumerate(paths):
        path = Path(path)
        suffixes.add(path.suffix)
        if not dry_run:
            os.symlink(path, Path(tempdir) / f"{i:0{padding}d}{path.suffix}")

    if len(suffixes) > 1:
        raise ValueError(f"Multiple file suffixes found in input paths: {suffixes}")
    return Path(tempdir), padding, suffixes.pop()


def do_ffmpeg(timelapse_cmd_list, expected_playback_time, no_progress=False):
    logger.debug(f"Running: {' '.join(timelapse_cmd_list)}")
    cmd = subprocess.Popen(
        timelapse_cmd_list, universal_newlines=True, stderr=subprocess.PIPE
    )
    progress = tqdm(
        total=expected_playback_time.total_seconds(),
        unit="output second",
        disable=no_progress,
    )
    # Examine each line of ffmpeg output (it all goes to stderr, for some reason):
    for line in cmd.stderr:
        # Lines that start with 'frame' are the "progress" lines
        if line.startswith("frame"):
            match = FFMPEG_PROGRESS_REGEX.match(line)
            if match:
                hours, minutes, seconds = match.groups()[0].split(":")
                hours = int(hours)
                minutes = int(minutes)
                seconds = float(seconds)
                td = timedelta(hours=hours, minutes=minutes, seconds=seconds)
                progress.n = td.total_seconds()
                progress.refresh()
        else:
            logger.debug(line.rstrip())


def timelapse(
    files_to_process,
    expected_playback_time,
    input_fps=30,
    output_fps=30,
    output_path="./out.mp4",
    staging_path=Path("./staging"),
    no_progress=False,
    no_edit=False,
    dry_run=False,
):
    if no_edit:
        logger.debug("Skipping image editing...")
        files_to_make_symlinks_to = files_to_process
    else:
        if not dry_run:
            staging_path.mkdir(exist_ok=True, parents=True)
        else:
            logger.debug(f"Skipping creation of {staging_path} due to --dry-run")
        padding = 9
        skipped = edit_images(
            files_to_process,
            output_dir=staging_path,
            padding=padding,
            no_progress=no_progress,
            dry_run=dry_run
            # tz="ET"
        )
        files_to_make_symlinks_to = staging_path.iterdir()

    if dry_run:
        logger.debug(
            f"Would have skipped copying {len(skipped)} files (but nothing happened due to dry_run=True"
        )
    else:
        logger.debug(f"Skipped copying {len(skipped)} files")

    # ffmpeg is BAD at handling multiple input files. To get around this, we first create
    # a bunch of symlinks that are named in a way that ffmpeg understands
    tempdir, padding, suffix = make_symlinks(files_to_make_symlinks_to, dry_run=dry_run)
    if not dry_run:
        logger.debug(f"Made tempdir {tempdir}")
    timelapse_cmd_list = [
        "ffmpeg",
        "-framerate",
        str(input_fps),
        "-i",
        str(os.path.join(tempdir, f"%0{padding}d{suffix}")),
        "-filter:v",
        f"eq=saturation=1.8:contrast=1.03:brightness=0.02:gamma=0.9",
        "-c:v",
        "libx265",
        # webm
        # "libvpx",
        # Sharpen output
        # "-crf",
        # "4",
        # "-filter:v",
        # "tblend",
        # "-filter:v",
        # "minterpolate",
        # Set bitrate
        # "-b:v",
        # "2000K",
        "-r",
        str(output_fps),
        "-pix_fmt",
        "yuv420p",
        # Overwrite existing files
        "-y",
        str(output_path),
    ]
    if dry_run:
        logger.debug(f"Would execute: {shlex.quote(' '.join(timelapse_cmd_list))}")
    else:
        do_ffmpeg(timelapse_cmd_list, expected_playback_time, no_progress=no_progress)

    if not dry_run:
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
    parser.add_argument(
        "--timestamp-source", choices=("metadata", "filename"), default="filename"
    )
    parser.add_argument(
        "--time-format",
        default=THE_FORMAT,
        help="Format string sent to strptime (default: %(default)s)",
    )
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
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="No progress bars (useful for cron scripts, etc.)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        init_logging(logging.DEBUG)
    else:
        init_logging(logging.INFO)
    logger.debug("Determining files to process...")
    files_to_process, stats = get_files_to_process(
        args.input,
        start=args.start,
        end=args.end,
        extended_stats=args.extended_stats,
        timestamp_source=args.timestamp_source,
        time_format=args.time_format,
        no_progress=args.no_progress,
    )
    logger.debug("...done")

    # logger.debug(pformat(stats))
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
    timelapse(
        files_to_process,
        input_fps=input_fps,
        output_fps=args.playback_fps,
        output_path=args.output,
        expected_playback_time=stats["timelapse_playback_time"],
        no_progress=args.no_progress,
        dry_run=args.dry_run,
    )


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super(self.__class__, self).__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            # self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def init_logging(level):
    """Initialize logging"""
    logging.getLogger().setLevel(level)
    _logger = logging.getLogger(__name__)
    console_handler = TqdmLoggingHandler()
    # console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(console_handler)
    _logger.setLevel(level)


if __name__ == "__main__":
    main()
