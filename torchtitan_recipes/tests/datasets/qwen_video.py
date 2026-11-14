# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in Qwen timestamped prompts for explicitly sampled NumPy videos."""

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import torch

from torchtitan.components.data.multimodal.mm_datasets import tokenize_mm_text
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.tokenizer import MultiModalTokenizer

from torchtitan_recipes.tests.datasets.synthetic_multimodal import load_npy_video_frames


class QwenVideoVisionProcessor(VisionProcessor):
    """Qwen video preprocessing with a pixel budget across the padded clip."""

    @dataclass(kw_only=True, slots=True)
    class Config(VisionProcessor.Config):
        video_min_pixels: int
        video_max_pixels: int
        video_frame_max_pixels: int | None = None

        def __post_init__(self) -> None:
            VisionProcessor.Config.__post_init__(self)
            if self.video_resize is not None:
                raise ValueError("Use clip-wide pixel limits instead of video_resize")
            min_frame_pixels = (self.patch_size * self.spatial_merge_size) ** 2
            if (
                self.video_frame_max_pixels is not None
                and self.video_frame_max_pixels < min_frame_pixels
            ):
                raise ValueError("video_frame_max_pixels cannot fit one spatial grid")

    config: Config

    def resize_and_normalize_video(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize and normalize a clip, returning its unmerged patch grid."""
        import torchvision.transforms.v2.functional as TVF

        config = self.config
        temporal_patch_size = config.temporal_patch_size
        num_frames, height, width, _ = frames.shape
        if num_frames < temporal_patch_size:
            raise ValueError(
                f"video requires at least {temporal_patch_size} sampled frames; got {num_frames}"
            )
        factor = config.patch_size * config.spatial_merge_size
        if height < factor or width < factor:
            scale = max(factor / height, factor / width)
            height, width = int(height * scale), int(width * scale)
        if max(height, width) / min(height, width) > 200:
            raise ValueError("video aspect ratio must be at most 200")
        padded_frames = num_frames + (-num_frames % temporal_patch_size)
        max_pixels = config.video_max_pixels
        if config.video_frame_max_pixels is not None:
            max_pixels = min(max_pixels, padded_frames * config.video_frame_max_pixels)
        max_spatial_grids = max_pixels // (padded_frames * factor * factor)
        if max_spatial_grids < 1:
            raise ValueError("video_max_pixels cannot fit one padded temporal patch")
        resize_h = round(height / factor) * factor
        resize_w = round(width / factor) * factor
        if padded_frames * resize_h * resize_w < config.video_min_pixels:
            scale = math.sqrt(
                config.video_min_pixels / (padded_frames * height * width)
            )
            resize_h = math.ceil(height * scale / factor) * factor
            resize_w = math.ceil(width * scale / factor) * factor
        if padded_frames * resize_h * resize_w > max_pixels:
            scale = math.sqrt(padded_frames * height * width / max_pixels)
            resize_h = max(factor, math.floor(height / scale / factor) * factor)
            resize_w = max(factor, math.floor(width / scale / factor) * factor)
            # A minimum-size short side can require a smaller long side.
            if resize_h <= resize_w:
                resize_w = min(
                    resize_w, max_spatial_grids // (resize_h // factor) * factor
                )
            else:
                resize_h = min(
                    resize_h, max_spatial_grids // (resize_w // factor) * factor
                )
        pixels = frames.permute(0, 3, 1, 2)
        if pixels.shape[-2:] != (resize_h, resize_w):
            pixels = TVF.resize(
                pixels,
                [resize_h, resize_w],
                interpolation=TVF.InterpolationMode(config.image_interpolation_mode),
                antialias=True,
            )
        # Match HF's fused rescale/normalize arithmetic on uint8 inputs.
        pixels = TVF.normalize(
            pixels.to(torch.float32),
            torch.tensor(config.image_mean) * 255,
            torch.tensor(config.image_std) * 255,
            inplace=True,
        ).permute(0, 2, 3, 1)
        grid_thw = torch.tensor(
            [
                padded_frames // temporal_patch_size,
                resize_h // config.patch_size,
                resize_w // config.patch_size,
            ]
        )
        return pixels, grid_thw


def process_qwen_video_sample(
    sample: dict[str, Any],
    tokenizer: MultiModalTokenizer,
    vision_processor: VisionProcessor,
    *,
    video_root: Path,
) -> dict[str, Any] | None:
    """Expand HF-style markers and retain one media item per temporal group.

    ``text`` is the complete tokenization input, including the final EOS, and
    is already chat-templated if needed. ``videos`` lists ``path`` (uint8 THWC
    .npy) and one nondecreasing source ``frame_indices``
    entry per stored frame. ``images`` optionally lists image inputs in prompt
    order. Relative local image and video paths are resolved under ``video_root``.
    Video frames are already sampled and decoded.

    Video min/max pixels use whole-clip semantics. The maximum includes temporal
    padding and takes priority over the minimum resize target after spatial
    rounding. Image budgets retain the shared processor's spatial semantics.
    For presentation time, provide all of ``frame_pts``, ``time_base``, and
    ``time_origin_pts``. The optional ``timestamp_source="clip_relative_pts"``
    also selects this mode. Otherwise timestamps use indices and a positive FPS.
    """
    if not isinstance(vision_processor, QwenVideoVisionProcessor):
        raise ValueError("Timestamped video samples require QwenVideoVisionProcessor")
    temporal_patch_size = vision_processor.config.temporal_patch_size
    text = sample["text"]
    videos = sample["videos"]
    images = sample.get("images", [])
    start = cast(
        str,
        tokenizer.vision_start_token,  # pyrefly: ignore [missing-attribute]
    )
    end = cast(
        str,
        tokenizer.vision_end_token,  # pyrefly: ignore [missing-attribute]
    )
    video_token = cast(
        str,
        tokenizer.video_token,  # pyrefly: ignore [missing-attribute]
    )
    image_token = cast(
        str,
        tokenizer.image_token,  # pyrefly: ignore [missing-attribute]
    )
    for token, items in ((video_token, videos), (image_token, images)):
        if text.count(token) != len(items) or text.count(start + token + end) != len(
            items
        ):
            raise ValueError(
                "Each image/video input requires one wrapped prompt marker"
            )

    processed_videos: list[torch.Tensor] = []
    video_replacements: list[str] = []
    for video in videos:
        frames = load_npy_video_frames(video_root / video["path"])
        indices = video["frame_indices"]
        fps = video.get("fps")
        frame_times = None
        if (
            any(
                field in video
                for field in ("frame_pts", "time_base", "time_origin_pts")
            )
            or video.get("timestamp_source") == "clip_relative_pts"
        ):
            pts = video.get("frame_pts")
            time_base = video.get("time_base")
            origin = video.get("time_origin_pts")
            if (
                not isinstance(pts, list)
                or len(pts) != len(frames)
                or any(type(value) is not int for value in pts)
                or any(left >= right for left, right in zip(pts, pts[1:]))
                or type(origin) is not int
                or any(value < origin for value in pts)
                or not isinstance(time_base, list)
                or len(time_base) != 2
                or any(type(value) is not int or value <= 0 for value in time_base)
            ):
                raise ValueError(
                    "PTS mode requires increasing integer frame_pts, a positive "
                    "rational time_base, and an integer time_origin_pts at or "
                    "before the first selected PTS"
                )
            scale = Fraction(*time_base)
            frame_times = [(value - origin) * scale for value in pts]
        else:
            if fps is None or not math.isfinite(fps) or fps <= 0:
                raise ValueError("video fps must be finite and positive")
        if (
            len(indices) != len(frames)
            or any(type(index) is not int or index < 0 for index in indices)
            or any(left > right for left, right in zip(indices, indices[1:]))
        ):
            raise ValueError(
                "frame_indices must match stored frames and be nonnegative and nondecreasing"
            )
        pixels, grid_thw = vision_processor.resize_and_normalize_video(frames)
        _, height, width = grid_thw.tolist()
        tokens_per_group = vision_processor.num_placeholder_tokens((1, height, width))
        num_frames = len(frames)
        groups = list(pixels.split(temporal_patch_size))
        replacement: list[str] = []
        for group_index, group in enumerate(groups):
            first = group_index * temporal_patch_size
            last = min(first + temporal_patch_size, num_frames) - 1
            if frame_times is None:
                timestamp = (indices[first] / fps + indices[last] / fps) / 2
            else:
                timestamp = float((frame_times[first] + frame_times[last]) / 2)
            replacement.append(
                f"<{timestamp:.1f} seconds>"
                + start
                + video_token * tokens_per_group
                + end
            )
            processed_videos.append(group)
        video_replacements.append("".join(replacement))
    # Modern Qwen3VLProcessor replaces the bare token, retaining the chat template wrappers.
    parts = text.split(video_token)
    text = parts[0] + "".join(
        value + part for value, part in zip(video_replacements, parts[1:])
    )

    processed_images: list[torch.Tensor] = []
    image_replacements: list[str] = []
    for image in images:
        if isinstance(image, str) and not image.startswith(("http://", "https://")):
            image = str(video_root / image)
        pixels, grid_thw = vision_processor.resize_and_normalize(image)
        count = vision_processor.num_placeholder_tokens(grid_thw)
        image_replacements.append(start + image_token * count + end)
        processed_images.append(pixels)
    parts = text.split(start + image_token + end)
    text = parts[0] + "".join(
        value + part for value, part in zip(image_replacements, parts[1:])
    )
    text_fields = tokenize_mm_text(text, tokenizer)
    if text_fields is None:
        return None
    return {
        **text_fields,
        "pixel_values": processed_images,
        "pixel_values_videos": processed_videos,
    }
