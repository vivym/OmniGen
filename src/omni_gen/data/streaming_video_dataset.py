import io
import math
import random

import av
import numpy as np
import torch
from streaming import Stream, StreamingDataset
from torchvision import transforms as T
from torchvision.transforms import _transforms_video as TrV

from .video_dataset import _frame_to_stamp, ShortSideScale


class StreamingVideoDataset(StreamingDataset):
    def __init__(
        self,
        streams: list[tuple[str, float]],
        spatial_size: int | tuple[int, int] = 256,
        num_frames: int = 17,
        frame_intervals: int | tuple[int, ...] = 1,
        training: bool = True,
        local_cache_dir: str | None = None,
        local_cache_limit: str | None = "100gb",
        shuffle_algo: str = "py1e",
        shuffle_seed: int = 233,
    ):
        total_weight = sum(weight for _, weight in streams)

        streams = [
            Stream(
                remote=path,
                local=local_cache_dir,
                proportion=weight / total_weight,
            )
            for path, weight in streams
        ]

        super().__init__(
            streams=streams,
            cache_limit=local_cache_limit,
            shuffle=training,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
            batching_method="per_stream",
        )

        self.spatial_size = spatial_size
        self.num_frames = num_frames
        if isinstance(frame_intervals, int):
            frame_intervals = (frame_intervals,)
        self.frame_intervals = tuple(frame_intervals)
        self.training = training

        self.transform = T.Compose([
            TrV.ToTensorVideo(),
            ShortSideScale(size=spatial_size, random_scale=True),
            TrV.RandomCropVideo(spatial_size) if training else TrV.CenterCropVideo(spatial_size),
            TrV.RandomHorizontalFlipVideo(p=0.5) if training else T.Lambda(lambda x: x),
        ])

    def __getitem__(self, idx: int):
        obj = super().__getitem__(idx)

        video_buf = io.BytesIO(obj["video"])
        container = av.open(video_buf)
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
            container = av.open(video_buf)
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
            raise ValueError(f"Failed to extract frames from #{obj['id']} frame_interval: {frame_interval} start_frame: {start_frame}")

        if len(frames) < self.num_frames:
            for _ in range(self.num_frames - len(frames)):
                frames.append(frames[-1])

        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames)
        frames = self.transform(frames)
        frames = frames * 2 - 1

        return {"pixel_values": obj["video"]}
