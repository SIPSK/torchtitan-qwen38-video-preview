# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import grain.python as grain
import numpy as np
import pytest
import torch
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from torchtitan.components.data import GrainDataLoader
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.multimodal import mm_datasets
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.utils.audio import load_pcm_waveform
from torchtitan.components.data.multimodal.vision_processor import (
    PixelBudgetResize,
    ResizeRule,
    VisionProcessor,
)
from torchtitan.components.data.sources import IndexedJsonlSource
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan_recipes.tests.datasets import synthetic_multimodal
from torchtitan_recipes.tests.datasets.qwen_video import (
    process_qwen_video_sample,
    QwenVideoVisionProcessor,
)


_ASSET_ROOT = Path(__file__).resolve().parents[4] / "assets/synthetic_multimodal"
_TOKEN_STRINGS = {
    "image": "<image>",
    "video": "<video>",
    "vision_start": "<vision_start>",
    "vision_end": "<vision_end>",
    "pad": "<pad>",
    "audio": "<audio>",
    "audio_start": "<audio_start>",
    "audio_end": "<audio_end>",
}


class _AudioMultiModalTokenizer(MultiModalTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(MultiModalTokenizer.Config):
        audio_token: str
        audio_start_token: str
        audio_end_token: str

    TOKEN_FIELDS = (
        *MultiModalTokenizer.TOKEN_FIELDS,
        "audio",
        "audio_start",
        "audio_end",
    )
    LOSS_MASK_TOKEN_FIELDS = (
        *MultiModalTokenizer.LOSS_MASK_TOKEN_FIELDS,
        "audio",
        "audio_start",
        "audio_end",
    )


@pytest.fixture
def audio_tokenizer(tmp_path):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "000": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(
        [AddedToken(token, special=True) for token in _TOKEN_STRINGS.values()]
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<eos>"}))
    tokenizer.add_special_tokens([AddedToken("<eos>", special=True)])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    config = _AudioMultiModalTokenizer.Config(
        **{f"{name}_token": token for name, token in _TOKEN_STRINGS.items()}
    )
    return _AudioMultiModalTokenizer(config, tokenizer_path=str(tmp_path))


def _first_manifest_row(modality: str) -> dict:
    manifest = _ASSET_ROOT / f"{modality}_text/data.jsonl"
    return json.loads(manifest.read_text().splitlines()[0])


def _assert_media_targets_masked(result, tokenizer, fields):
    target_ids = result["input_ids"][1:]
    labels = result["labels"][:-1]
    media_token_ids = torch.tensor(
        [getattr(tokenizer, f"{field}_id") for field in fields]
    )
    media_targets = torch.isin(target_ids, media_token_ids)
    assert torch.any(media_targets)
    assert torch.all(labels[media_targets] == IGNORE_INDEX)


def test_audio_sample_processor_resamples_truncates_and_masks(audio_tokenizer):
    row = _first_manifest_row("audio")
    waveform, sample_rate = load_pcm_waveform(
        _ASSET_ROOT / "audio_text" / row["audio_path"],
        expected_num_channels=1,
    )

    result = mm_datasets._process_audio_text_sample(
        text=row["text"],
        waveform=waveform,
        source_sample_rate=sample_rate,
        tokenizer=audio_tokenizer,
        target_sample_rate=16_000,
        max_audio_samples=400,
        audio_token_count_fn=lambda num_samples, sample_rate: 5,
    )

    assert result["waveforms"][0].shape == (400, 1)
    assert result["waveforms"][0].dtype == torch.float32
    assert int((result["input_ids"] == audio_tokenizer.audio_id).sum()) == 5
    _assert_media_targets_masked(
        result,
        audio_tokenizer,
        ("audio", "audio_start", "audio_end"),
    )


def test_load_npy_video_frames_returns_c_contiguous_uint8_thwc(tmp_path):
    frames_THWC = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    path = tmp_path / "frames.npy"
    np.save(path, frames_THWC)

    actual_THWC = synthetic_multimodal.load_npy_video_frames(path)

    assert actual_THWC.is_contiguous()
    assert torch.equal(actual_THWC, torch.from_numpy(frames_THWC))


class _PadWidthResize(ResizeRule):
    """Keeps each frame's size and pads its width by 6 pixels."""

    @dataclass(kw_only=True, slots=True)
    class Config(ResizeRule.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def __call__(
        self, height: int, width: int, *, patch_size: int, merge_size: int
    ) -> tuple[int, int, int, int]:
        return height, width, 0, 6


def test_video_sample_processor_resizes_and_masks(audio_tokenizer):
    row = _first_manifest_row("video")
    frames = synthetic_multimodal.load_npy_video_frames(
        _ASSET_ROOT / "video_text" / row["video_path"]
    )

    vision_processor = VisionProcessor.Config(
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=1,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        resize=_PadWidthResize.Config(),
        patch_order="block",
        video_resize=_PadWidthResize.Config(),
    ).build()

    result = mm_datasets._process_video_text_sample(
        text=row["text"],
        frames=frames,
        tokenizer=audio_tokenizer,
        vision_processor=vision_processor,
    )

    # Two 28 x 36 frames padded to 28 x 42.
    assert result["pixel_values_videos"][0].shape == (2, 28, 42, 3)
    assert torch.any(result["input_ids"] == audio_tokenizer.video_id)
    _assert_media_targets_masked(
        result,
        audio_tokenizer,
        ("video", "vision_start", "vision_end"),
    )


def test_synthetic_datasets_register_absolute_sources_and_processors():
    expected = {
        "synthetic-audio-text": ("audio_text", mm_datasets.AudioProcessor.Config),
        "synthetic-video-text": (
            "video_text",
            mm_datasets.VisionSampleProcessor.Config,
        ),
    }
    for name, (directory, processor_type) in expected.items():
        dataset = synthetic_multimodal.SYNTHETIC_MM_DATASETS[name]
        assert isinstance(dataset, SingleDatasetConfig)
        assert isinstance(dataset.source, IndexedJsonlSource.Config)
        assert dataset.source.patterns == (
            str((_ASSET_ROOT / directory / "data.jsonl").resolve()),
        )
        assert isinstance(dataset.processor, processor_type)


def _synthetic_loader(audio_tokenizer, dataset_name: str):
    dataset = synthetic_multimodal.SYNTHETIC_MM_DATASETS[dataset_name]
    vision_processor = None
    if dataset_name == "synthetic-video-text":
        resize = PixelBudgetResize.Config(min_pixels=784, max_pixels=200_000)
        vision_processor = VisionProcessor.Config(
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            resize=resize,
            patch_order="block",
            video_resize=resize,
        ).build()
    collator = MultiModalCollator.Config()
    if dataset_name == "synthetic-audio-text":
        collator = replace(
            collator,
            waveform_pad_multiple=4,
            audio_input_channels=1,
        )
    return GrainDataLoader.Config(
        dataset=dataset,
        collator=collator,
        shuffle=True,
        repeat=True,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
        num_prefetch_microbatches=1,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=audio_tokenizer,
        max_context_length=512,
        num_tokens_per_microbatch=512,
        vision_processor=vision_processor,
    )


@pytest.mark.parametrize(
    "dataset_name",
    ["synthetic-audio-text", "synthetic-video-text"],
)
def test_synthetic_loader_restores_exact_next_batch(audio_tokenizer, dataset_name):
    loader = _synthetic_loader(audio_tokenizer, dataset_name)
    iterator = iter(loader)
    try:
        next(iterator)
        state = loader.state_dict()
        expected = next(iterator).as_input_dict()

        resumed = _synthetic_loader(audio_tokenizer, dataset_name)
        try:
            resumed.load_state_dict(state)
            actual = next(iter(resumed)).as_input_dict()
        finally:
            resumed.close()
    finally:
        loader.close()

    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, torch.Tensor):
            assert torch.equal(actual_value, expected_value), key
        else:
            assert actual_value == expected_value, key
    media_key = (
        "waveforms" if dataset_name == "synthetic-audio-text" else "pixel_values_videos"
    )
    assert actual[media_key].numel() > 0


def _collator(audio_tokenizer):
    context = DatasetBuildContext(
        tokenizer=audio_tokenizer,
        max_context_length=16,
        num_tokens_per_microbatch=16,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
    )
    return MultiModalCollator.Config(
        waveform_pad_multiple=4,
        audio_input_channels=1,
    ).build(context=context)


def _collator_row(audio_tokenizer, waveforms):
    tokens = torch.tensor(
        [
            audio_tokenizer.audio_start_id,
            audio_tokenizer.audio_id,
            audio_tokenizer.audio_end_id,
        ]
    )
    return {
        "input_ids": tokens,
        "labels": tokens.clone(),
        "positions": torch.arange(len(tokens)),
        "pixel_values": [],
        "pixel_values_videos": [],
        "waveforms": waveforms,
    }


def test_audio_collator_pads_waveforms_in_clip_order(audio_tokenizer):
    first = torch.tensor([[0.1], [0.2], [0.3]])
    second = torch.tensor([[-0.1], [-0.2], [-0.3], [-0.4], [-0.5]])

    batch = _collator(audio_tokenizer)(
        [_collator_row(audio_tokenizer, [first, second])]
    )

    assert batch.model_kwargs["waveform_lengths"].tolist() == [3, 5]
    assert batch.model_kwargs["waveform_padded_lengths"].tolist() == [4, 8]
    packed = batch.model_kwargs["waveforms"]
    assert torch.equal(packed[:3], first)
    assert torch.equal(packed[4:9], second)


def test_audio_collator_omits_audio_kwargs_without_waveforms(audio_tokenizer):
    batch = _collator(audio_tokenizer)([_collator_row(audio_tokenizer, [])])

    assert (
        not {
            "waveforms",
            "waveform_lengths",
            "waveform_padded_lengths",
        }
        & batch.model_kwargs.keys()
    )


def _qwen_video_context(
    tokenizer,
    *,
    video_min_pixels=1,
    video_max_pixels=1_000_000,
    video_frame_max_pixels=None,
    max_context_length=256,
):
    vision_processor = QwenVideoVisionProcessor.Config(
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        resize=PixelBudgetResize.Config(min_pixels=1, max_pixels=1_000_000),
        patch_order="block",
        video_min_pixels=video_min_pixels,
        video_max_pixels=video_max_pixels,
        video_frame_max_pixels=video_frame_max_pixels,
    ).build()
    return DatasetBuildContext(
        tokenizer=tokenizer,
        vision_processor=vision_processor,
        max_context_length=max_context_length,
        num_tokens_per_microbatch=max_context_length,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
    )


def _qwen_video_processor(tmp_path, context):
    return mm_datasets.VisionSampleProcessor.Config(
        sample_processor=partial(process_qwen_video_sample, video_root=tmp_path)
    ).build(context=context)


@pytest.mark.parametrize(
    (
        "num_frames",
        "height",
        "width",
        "min_pixels",
        "max_pixels",
        "frame_max_pixels",
        "expected_size",
    ),
    [
        (5, 32, 64, 1, 8192, None, (32, 32)),
        (5, 512, 512, 1, 1_048_576, None, (416, 416)),
        (2, 32, 6400, 1, 2048, None, (32, 32)),
        (2, 6400, 32, 1, 2048, None, (32, 32)),
        (2, 32, 32, 8000, 8000, None, (32, 32)),
        (5, 32, 32, 1, 6144, None, (32, 32)),
        (5, 32, 32, 1, 6143, None, None),
        (4, 512, 512, 1, 1_048_576, None, (512, 512)),
        (2, 32, 32, 8192, 8192, None, (64, 64)),
        (5, 512, 512, 1, 1_048_576, 65_536, (256, 256)),
        (2, 64, 64, 4096, 8192, 1024, (32, 32)),
    ],
)
def test_qwen_timestamped_video_budget_includes_temporal_padding(
    tmp_path,
    audio_tokenizer,
    num_frames,
    height,
    width,
    min_pixels,
    max_pixels,
    frame_max_pixels,
    expected_size,
):
    np.save(
        tmp_path / "video.npy", np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    )
    sample = {
        "text": "<vision_start><video><vision_end>caption<eos>",
        "videos": [
            {"path": "video.npy", "fps": 2, "frame_indices": list(range(num_frames))}
        ],
    }
    context = _qwen_video_context(
        audio_tokenizer,
        video_min_pixels=min_pixels,
        video_max_pixels=max_pixels,
        video_frame_max_pixels=frame_max_pixels,
        max_context_length=2048,
    )
    processor = _qwen_video_processor(tmp_path, context)
    if expected_size is None:
        with pytest.raises(ValueError, match="video_max_pixels"):
            processor(sample, np.random.default_rng(0))
        return

    row = processor(sample, np.random.default_rng(0))
    groups = row["pixel_values_videos"]
    assert all(tuple(group.shape[1:3]) == expected_size for group in groups)
    # Count actual pixels after the same temporal padding used by the collator.
    patches = [context.vision_processor.patchify(group)[0] for group in groups]
    assert sum(patch.numel() // 3 for patch in patches) <= max_pixels


def test_qwen_timestamped_video_rejects_frame_budget_below_grid(audio_tokenizer):
    with pytest.raises(ValueError, match="video_frame_max_pixels"):
        _qwen_video_context(audio_tokenizer, video_frame_max_pixels=1023)


@pytest.mark.parametrize(
    ("frame_indices", "fps", "timestamps"),
    [
        ([0, 3], 6.0, ["0.2"]),
        ([0, 3, 9], 6.0, ["0.2", "1.5"]),
        ([0, 3, 9, 15, 18], 6.0, ["0.2", "2.0", "3.0"]),
        ([1, 2], 2.0, ["0.8"]),
        ([0, 7], 10.0, ["0.3"]),
        ([0], 6.0, None),
    ],
)
def test_qwen_timestamped_video_retains_temporal_patches_and_shift(
    tmp_path, audio_tokenizer, frame_indices, fps, timestamps
):
    num_frames = len(frame_indices)
    frame_values = 30 * torch.arange(num_frames, dtype=torch.uint8)
    frames = frame_values[:, None, None, None].expand(-1, 32, 64, 3).contiguous()
    np.save(tmp_path / "video.npy", frames.numpy())
    sample = {
        "text": "before<vision_start><video><vision_end>after<eos>",
        "videos": [
            {
                "path": "video.npy",
                "fps": fps,
                "frame_indices": frame_indices,
            }
        ],
    }
    context = _qwen_video_context(audio_tokenizer)
    processor = _qwen_video_processor(tmp_path, context)
    if timestamps is None:
        with pytest.raises(ValueError, match="at least 2 sampled frames.*got 1"):
            processor(sample, np.random.default_rng(0))
        return
    result = processor(sample, np.random.default_rng(0))
    expected_text = (
        "before<vision_start>"
        + "".join(
            f"<{time} seconds><vision_start><video><video><vision_end>"
            for time in timestamps
        )
        + "<vision_end>after<eos>"
    )
    expected_ids = torch.tensor(audio_tokenizer.encode(expected_text))
    assert torch.equal(result["input_ids"], expected_ids[:-1])
    expected_labels = expected_ids[1:].clone()
    expected_labels[
        torch.isin(
            expected_labels,
            torch.tensor(
                [
                    getattr(audio_tokenizer, f"{field}_id")
                    for field in audio_tokenizer.LOSS_MASK_TOKEN_FIELDS
                ]
            ),
        )
    ] = IGNORE_INDEX
    assert torch.equal(result["labels"], expected_labels)
    groups = result["pixel_values_videos"]
    assert [len(group) for group in groups] == [2] * (num_frames // 2) + (
        [1] if num_frames % 2 else []
    )
    whole, whole_grid = context.vision_processor.patchify(torch.cat(groups))
    split = [context.vision_processor.patchify(group) for group in groups]
    assert torch.equal(whole, torch.cat([patches for patches, _ in split]))
    assert [grid.tolist() for _, grid in split] == [[1, 2, 4]] * len(timestamps)
    assert whole_grid.tolist() == [len(timestamps), 2, 4]

    normalized_frames = (frame_values.float() - 127.5) / 127.5
    for group_index, (patches, _) in enumerate(split):
        first = 2 * group_index
        second = min(first + 1, num_frames - 1)
        # Each flattened patch is channel-first, then temporal, height, width.
        temporal_patches = patches.reshape(-1, 3, 2, 16, 16)
        expected = normalized_frames[[first, second]].view(1, 1, 2, 1, 1)
        torch.testing.assert_close(
            temporal_patches, expected.expand_as(temporal_patches)
        )


def test_qwen_timestamped_video_keeps_distinct_one_token_groups(
    tmp_path, audio_tokenizer
):
    from PIL import Image
    from torchtitan.models.common.multimodal import get_vision_positions

    np.save(tmp_path / "first.npy", np.full((3, 32, 32, 3), 30, dtype=np.uint8))
    np.save(tmp_path / "second.npy", np.full((2, 32, 64, 3), 220, dtype=np.uint8))
    Image.new("RGB", (32, 32), (100, 100, 100)).save(tmp_path / "still.png")
    marker = "<vision_start><video><vision_end>"
    sample = {
        "text": "left"
        + marker
        + "middle<vision_start><image><vision_end>between"
        + marker
        + "tail<eos>",
        "videos": [
            {"path": "first.npy", "fps": 2, "frame_indices": [0, 2, 4]},
            {"path": "second.npy", "fps": 10, "frame_indices": [20, 40]},
        ],
        "images": ["still.png"],
    }
    context = _qwen_video_context(audio_tokenizer)
    processor = _qwen_video_processor(tmp_path, context)
    row = processor(sample, np.random.default_rng(0))
    batch = (
        MultiModalCollator.Config(build_mrope_positions=True)
        .build(context=context)([row])
        .as_input_dict()
    )
    assert batch["grid_thw_videos"].tolist() == [[1, 2, 2], [1, 2, 2], [1, 2, 4]]
    assert batch["grid_thw"].tolist() == [[1, 2, 2]]
    spans = get_vision_positions(
        batch["input"], torch.tensor([1, 1, 2]), audio_tokenizer.video_id
    )
    assert [length for _, _, length in spans] == [1, 1, 2]
    patches = batch["pixel_values_videos"]
    assert torch.all(patches[:8] < 0)
    assert torch.all(patches[8:] > 0)
    expected_mrope = batch["positions"].unsqueeze(-1).expand(-1, 3).clone()
    last_start = spans[-1][1]
    expected_mrope[last_start + 1, :2] = last_start
    assert torch.equal(batch["mrope_positions"], expected_mrope)
    bad_marker = {**sample, "text": sample["text"].replace(marker, "<video>", 1)}
    with pytest.raises(ValueError, match="one wrapped prompt marker"):
        processor(bad_marker, np.random.default_rng(0))

    bad = {
        **sample,
        "videos": [
            {**sample["videos"][0], "frame_indices": [2, 0, 4]},
            sample["videos"][1],
        ],
    }
    with pytest.raises(ValueError, match="frame_indices"):
        processor(bad, np.random.default_rng(0))
    with pytest.raises(ValueError, match="clip-wide pixel limits"):
        replace(
            context.vision_processor.config,
            video_resize=PixelBudgetResize.Config(min_pixels=1, max_pixels=1024),
        )
    with pytest.raises(ValueError, match="require QwenVideoVisionProcessor"):
        process_qwen_video_sample(
            sample,
            audio_tokenizer,
            VisionProcessor(context.vision_processor.config),
            video_root=tmp_path,
        )


def test_qwen_timestamped_video_packing_restores_exact_next_batch(
    tmp_path, audio_tokenizer, monkeypatch
):
    from torchtitan.config import ConfigLoader

    monkeypatch.chdir(tmp_path)
    video_root = tmp_path / "assets/qwen38-video"
    video_root.mkdir(parents=True)
    (tmp_path / "assets/hf/Qwen3.8-27B").mkdir(parents=True)
    manifest = video_root / "manifest.jsonl"
    rows = []
    for i in range(8):
        path = f"video-{i}.npy"
        np.save(video_root / path, np.full((3, 64, 64, 3), i * 30, dtype=np.uint8))
        rows.append(
            {
                "text": f"row {i}<vision_start><video><vision_end>caption {i}<eos>",
                "videos": [{"path": path, "fps": 2, "frame_indices": [0, 2, 4]}],
            }
        )
    manifest.write_text("\n".join(json.dumps(row) for row in rows))

    def build_loader():
        config = ConfigLoader().load(
            [
                "--module",
                "torchtitan_recipes.tests.models.qwen3_8",
                "--config",
                "qwen38_debugmodel_timestamped_video",
            ]
        )
        return replace(
            config.dataloader,
            dataset=mm_datasets.MMSamplePackingConfig(
                dataset=config.dataloader.dataset, num_packing_bins=2
            ),
            shuffle=False,
            repeat=True,
            read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
            num_prefetch_microbatches=1,
        ).build(
            dp_world_size=1,
            dp_rank=0,
            tokenizer=audio_tokenizer,
            max_context_length=64,
            num_tokens_per_microbatch=128,
            vision_processor=config.vision_processor.build(),
        )

    loader = build_loader()
    iterator = iter(loader)
    try:
        first = next(iterator).as_input_dict()
        state = loader.state_dict()
        expected = next(iterator).as_input_dict()
        assert not torch.equal(
            first["pixel_values_videos"], expected["pixel_values_videos"]
        )
        resumed = build_loader()
        try:
            resumed.load_state_dict(state)
            actual = next(iter(resumed)).as_input_dict()
        finally:
            resumed.close()
    finally:
        loader.close()
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(actual[key], value), key
        else:
            assert actual[key] == value, key
    active = ~actual["padding_mask"]
    tokens = actual["input"][active]
    mrope = actual["mrope_positions"][active]
    starts = torch.where(actual["positions"][active] == 0)[0].tolist()
    assert len(starts) >= 2
    assert actual["grid_thw_videos"].tolist() == [[1, 4, 4]] * (2 * len(starts))
    for start, end in zip(starts, starts[1:] + [len(tokens)], strict=True):
        assert mrope[start].tolist() == [0, 0, 0]
        video_tokens = torch.where(tokens[start:end] == audio_tokenizer.video_id)[0]
        assert len(video_tokens) == 8
        for group, first_token in enumerate(video_tokens[::4].tolist()):
            # A 2x2 merged grid occupies four tokens and advances M-RoPE by two.
            base = first_token - 2 * group
            offset = start + first_token
            assert mrope[offset : offset + 4].tolist() == [
                [base, base, base],
                [base, base, base + 1],
                [base, base + 1, base],
                [base, base + 1, base + 1],
            ]
        assert mrope[end - 1].tolist() == [end - start - 5] * 3


def test_qwen_timestamped_video_uses_presentation_time(
    tmp_path, audio_tokenizer, monkeypatch
):
    np.save(tmp_path / "video.npy", np.zeros((5, 32, 32, 3), dtype=np.uint8))
    video = {
        "path": "video.npy",
        "frame_indices": [0, 1, 3, 4, 6],
        "frame_pts": [100, 150, 430, 800, 1900],
        "time_base": [1, 1000],
        "time_origin_pts": 100,
    }
    sample = {
        "text": "<vision_start><video><vision_end>caption<eos>",
        "videos": [video],
    }
    encoded_text = []
    encode = audio_tokenizer.encode

    def record_text(text):
        encoded_text.append(text)
        return encode(text)

    monkeypatch.setattr(audio_tokenizer, "encode", record_text)
    processor = _qwen_video_processor(tmp_path, _qwen_video_context(audio_tokenizer))
    row = processor(sample, np.random.default_rng(0))
    assert [len(group) for group in row["pixel_values_videos"]] == [2, 2, 1]
    assert encoded_text == [
        "<vision_start>"
        "<0.0 seconds><vision_start><video><vision_end>"
        "<0.5 seconds><vision_start><video><vision_end>"
        "<1.8 seconds><vision_start><video><vision_end>"
        "<vision_end>caption<eos>"
    ]
    for missing in ("time_base", "time_origin_pts"):
        incomplete = {key: value for key, value in video.items() if key != missing}
        with pytest.raises(ValueError, match="frame_pts"):
            processor({**sample, "videos": [incomplete]}, np.random.default_rng(0))

    marker_only = {
        "path": video["path"],
        "frame_indices": video["frame_indices"],
        "fps": 2.0,
        "timestamp_source": "clip_relative_pts",
    }
    with pytest.raises(ValueError, match="frame_pts"):
        processor({**sample, "videos": [marker_only]}, np.random.default_rng(0))
