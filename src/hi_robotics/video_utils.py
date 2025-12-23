import av
import av.container
import av.stream
import av.video.frame
from fractions import Fraction

import logging; logger = logging.getLogger(__name__)

# Set log level https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/master/Source/Lib/Codec/svt_log.h
import os; os.environ["SVT_LOG"] = "2"


def load_video_frame(
    video_path: str,
    time_s: float | None = None,
    frame_idx: int | None = None,
    fps: float | None = None,
    frame_format: str | None = "rgb24"
):
    """
    Use PyAV to load specific frame of a video and return in PIL Image format

    Args:
        video_path (str): video file path
        time_s (float): time in seconds
        frame_idx (int): frame index
        fps (float): frames per second of the video

    Returns:
        PIL Image: the frame image
    """
    container: av.container.InputContainer = av.open(video_path, "r")

    video_stream: av.stream.Stream = container.streams.video[0]

    try:
        if time_s is not None:
            assert time_s >= 0
            target_pts: int = round(time_s / video_stream.time_base) # TODO float accuracy?
        elif frame_idx is not None:
            assert frame_idx >= 0
            if fps is None:
                fps = video_stream.average_rate
                logger.warning(f"Using average_rate {fps}, may not accurate")
            target_pts: int = round(frame_idx / fps / video_stream.time_base) # TODO float accuracy?
        else:
            raise Exception("Provide time_s or frame_idx")

        container.seek(target_pts, backward=True, any_frame=False, stream=video_stream)

        last_frame = None

        # print(f"target_pts {target_pts}")
        for frame in container.decode(video_stream):
            # print(f"decoding frame.pts {frame.pts}")
            if frame.pts > target_pts: break
            last_frame = frame
            if frame.pts == target_pts: break

        assert last_frame is not None

        if frame_format == "PIL_IMAGE":
            return last_frame.to_image()
        elif frame_format is not None:
            return last_frame.to_ndarray(format=frame_format)
        else:
            return last_frame
    finally:
        container.close()


class AvVideoReader:
    def __init__(
        self,
        video_path: str,
        frame_format: str | None = "rgb24"
    ):
        self.container: av.container.InputContainer = av.open(video_path, mode="r")
        self.video_stream: av.stream.Stream = self.container.streams.video[0]
        self.frame_format: str = frame_format
        self.frame_iterator = self.container.decode(self.video_stream)

    @property
    def framerate(self):
        # Unreliable!
        # frames in mkv is 0
        return self.video_stream.frames / self.video_stream.duration / self.video_stream.time_base

    def __iter__(self):
        return self

    def __next__(self):
        frame: av.video.frame.VideoFrame = next(self.frame_iterator)
        return frame.to_ndarray(format=self.frame_format) if self.frame_format else frame

    def __len__(self):
        return self.video_stream.frames

    def close(self):
        self.container.close()


class AvVideoWriter:
    VIDEO_CLOCK_RATE = 90000

    def __init__(
        self,
        video_path: str,
        fps: float = 30,
        width: int = 1280,
        height: int = 720,
        vcodec: str = "libsvtav1",
        pix_fmt: str = "yuv420p",
        frame_format: str | None = "rgb24",
        options: dict[str, str] | None = None,
        use_tmp = False, # Support for hdfs
    ):
        self.fps: float = fps
        self.frame_format: str | None = frame_format

        if options is None:
            options = {
                "g": str(2), # GOP Size,
                "crf": str(30),
                "preset": str(10),
            }

        self.options: dict[str, str] = options

        if use_tmp:
            uuid = os.urandom(16).hex()
            from pathlib import Path
            self.tmp_file = f"/tmp/{uuid}.{Path(video_path).suffix}"
            self.video_path = video_path

            self.container: av.container.OutputContainer = av.open(file=self.tmp_file, mode="w")
        else:
            self.container: av.container.OutputContainer = av.open(file=video_path, mode="w")

        self.stream: av.stream.Stream = self.container.add_stream(vcodec, rate=self.fps, options=self.options)
        self.stream.pix_fmt = pix_fmt

        self.stream.width = width
        self.stream.height = height

        self.pts = 0

    def append(self, image):
        frame = av.video.frame.VideoFrame.from_ndarray(image, format=self.frame_format) if self.frame_format else image

        frame.pts = self.pts
        frame.time_base = Fraction(1, self.VIDEO_CLOCK_RATE)
        self.pts += int(self.VIDEO_CLOCK_RATE / self.fps)

        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode(None):
            self.container.mux(packet)

        self.container.close()

        if hasattr(self, "tmp_file"):
            import shutil
            shutil.move(self.tmp_file, self.video_path)
