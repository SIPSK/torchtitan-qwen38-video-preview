# Timestamped Qwen video samples

The `qwen38_debugmodel_timestamped_video` recipe reads
`./assets/qwen38-video/manifest.jsonl` and uses model assets under
`./assets/hf/Qwen3.8-27B`. To use different paths, call the factory with
`manifest`, `video_root`, and `hf_assets_path` in your recipe module.
The manifest contains rendered prompts with their final EOS and prepared
uint8 NumPy clips in `(T, H, W, 3)` layout.

```json
{"text":"<|vision_start|><|video_pad|><|vision_end|>Describe the clip.<|im_end|>","videos":[{"path":"clip.npy","fps":6.0,"frame_indices":[0,3,9]}]}
```

Each image or video in the row has one wrapped marker, in prompt order within
its modality. `images` is an optional list of image paths or URLs. Resolve
relative image and video paths under `video_root`, which defaults to the
manifest directory. Each video's
`frame_indices` is required in both timing modes and contains one
nonnegative, nondecreasing source index per stored frame.
For index-based timestamps, provide a positive `fps`. For clip-relative
presentation timestamps, provide all three fields together: strictly increasing
integer `frame_pts` (one per frame), `time_base` (positive integer numerator
and denominator), and integer `time_origin_pts` at or before the first selected
PTS. The optional `timestamp_source: "clip_relative_pts"` selects PTS mode
even if a timing field is missing, so incomplete metadata is reported.

`QwenVideoVisionProcessor` resizes and normalizes the whole clip. The budget
includes the frame padding used by patchification. The recipe sets
`video_min_pixels=4096` and `video_max_pixels=25165824`. Set the upper limit
on the processor config to fit the recipe's token context, including text and
any other media in the same row.
The optional `video_frame_max_pixels` limits the area of each resized frame.
It defaults to `None`. The processor applies the stricter of the frame and
padded clip limits before choosing a single resize.

The sample processor replaces the inner video marker with one timestamped
placeholder run per temporal group, retaining the template's outer markers.
The example has timestamps `0.2` and `1.5` seconds. It passes each group as a
separate video item. The collator patchifies each item to a `[1, H, W]` grid,
and the shared MRoPE implementation uses that grid for its placeholder run.
Clips require at least two sampled frames. The resize calculation scales
short sides up to the minimum spatial grid before applying the pixel budget.
For an odd final group, patchification repeats its last frame. Each image
and each video time group count toward the collator's limit of 128 vision
entries per microbatch, including entries from different packed documents.
