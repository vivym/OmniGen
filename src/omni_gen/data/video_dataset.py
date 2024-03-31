import math
import random

import av
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import _transforms_video as TrV


def _frame_to_stamp(nframe, stream) -> int:
    """Convert frame number to timestamp based on fps of video stream."""
    fps = stream.guessed_rate.numerator / stream.guessed_rate.denominator
    seek_target = nframe / fps
    stamp = math.floor(
        seek_target * (stream.time_base.denominator / stream.time_base.numerator)
    )
    return stamp


def short_side_scale(
    x: torch.Tensor,
    size: int,
    interpolation: str = "bilinear",
) -> torch.Tensor:
    """
    Determines the shorter spatial dim of the video (i.e. width or height) and scales
    it to the given size. To maintain aspect ratio, the longer side is then scaled
    accordingly.
    Args:
        x (torch.Tensor): A video tensor of shape (C, T, H, W) and type torch.float32.
        size (int): The size the shorter side is scaled to.
        interpolation (str): Algorithm used for upsampling,
            options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
    Returns:
        An x-like Tensor with scaled spatial dims.
    """  # noqa
    assert len(x.shape) == 4
    assert x.dtype == torch.float32
    _, _, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))

    return torch.nn.functional.interpolate(
        x, size=(new_h, new_w), mode=interpolation, align_corners=False
    )


class ShortSideScale:
    def __init__(self, size: int, interpolation: str = "bilinear"):
        self._size = size
        self._interpolation = interpolation

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return short_side_scale(x, self._size, self._interpolation)


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
            TrV.ToTensorVideo(),
            ShortSideScale(size=spatial_size),
            TrV.RandomCropVideo(spatial_size) if training else TrV.CenterCropVideo(spatial_size),
            TrV.RandomHorizontalFlipVideo(p=0.5) if training else T.Lambda(lambda x: x),
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

        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames)
        frames = self.transform(frames)
        frames = frames * 2 - 1

        return {"pixel_values": frames}
