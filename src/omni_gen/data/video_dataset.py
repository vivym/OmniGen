import math
import random

import av
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


def _frame_to_stamp(nframe, stream) -> int:
    """Convert frame number to timestamp based on fps of video stream."""
    fps = stream.guessed_rate.numerator / stream.guessed_rate.denominator
    seek_target = nframe / fps
    stamp = math.floor(
        seek_target * (stream.time_base.denominator / stream.time_base.numerator)
    )
    return stamp


class VideoDataset(Dataset):
    def __init__(
        self,
        video_paths: list[str],
        spatial_size: int | tuple[int, int] = 256,
        num_frames: int = 17,
        frame_intervals: int | tuple[int, ...] = 1,
        training: bool = True,
    ):
        if isinstance(frame_intervals, int):
            frame_intervals = (frame_intervals,)

        self.video_paths = video_paths
        self.spatial_size = spatial_size
        self.num_frames = num_frames
        self.frame_intervals = tuple(frame_intervals)
        self.training = training

        self.transform = T.Compose([
            T.Resize(spatial_size),
            T.RandomCrop(spatial_size) if training else T.CenterCrop(spatial_size),
            T.RandomHorizontalFlip() if training else T.Lambda(lambda x: x),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        video_path = self.video_paths[idx]

        container = av.open(video_path)
        video_stream = container.streams.video[0]
        total_frames = video_stream.frames

        if total_frames == 0:
            duration = video_stream.duration

            if duration is None:
                duration = container.duration
                time_base = 1 / av.time_base
            else:
                time_base = video_stream.time_base.numerator / video_stream.time_base.denominator

            fps = video_stream.guessed_rate.numerator / video_stream.guessed_rate.denominator

            total_frames = math.floor(fps * duration * time_base)

        frame_intervals = [
            interval
            for interval in self.frame_intervals
            if self.num_frames * interval <= total_frames
        ]
        if len(frame_intervals) == 0:
            frame_interval = 1
        else:
            frame_interval = random.choice(frame_intervals)

        gop_size = (
            video_stream.codec_context.gop_size
        )  # gop size is distance (in frames) between 2 I-frames
        if frame_interval > gop_size:
            step_seeking = True
        else:
            step_seeking = False

        start_frame = random.randint(0, max(total_frames - self.num_frames * frame_interval, 0))

        seek_target = _frame_to_stamp(start_frame, video_stream)
        step_time = _frame_to_stamp(frame_interval, video_stream)

        seekable = True
        try:
            container.seek(seek_target, stream=video_stream)
        except av.error.FFmpegError:
            seekable = False
            # try again but this time don't seek
            container = av.open(video_path)
            video_stream = container.streams.video[0]

        frames = []
        for packet in container.demux(video=0):
            if len(frames) >= self.num_frames or packet.dts is None:
                continue

            for frame in packet.decode():
                if packet.pts and packet.pts >= seek_target:
                    frame = frame.to_ndarray(format="rgb24")
                    frame = Image.fromarray(frame)
                    frame = self.transform(frame)
                    frames.append(frame)
                    if len(frames) >= self.num_frames:
                        break

                    seek_target += step_time
                    if step_seeking and seekable:
                        container.seek(seek_target, stream=video_stream)

        if len(frames) == 0:
            raise ValueError(f"Failed to extract frames from {video_path} frame_interval: {frame_interval} start_frame: {start_frame}")

        if len(frames) < self.num_frames:
            for _ in range(self.num_frames - len(frames)):
                frames.append(frames[-1])

        frames = torch.stack(frames, dim=1)

        frames = frames * 2 - 1

        return {"pixel_values": frames}
