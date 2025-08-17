# TransNetV2 PyTorch Inference (Batch)

This script runs **scene boundary prediction** on a single video or a folder of multiple videos using the PyTorch implementation of [TransNetV2.](https://github.com/soCzech/TransNetV2)

## Requirements

* PyTorch
* ffmpeg (command-line tool)
* `ffmpeg-python` (`pip install ffmpeg-python`)
* `Pillow`

```sh
apt-get install ffmpeg
python -m venv .venv
source .venv/bin/activate
pip install ffmpeg-python torch pillow
```

Or use NVIDIA Docker:
```sh
# build
docker build -t transnetv2_pytorch .

# run on a folder
docker run --rm --gpus all \
  -v /my/local/videos:/data \
  transnetv2_pytorch /data
```

Make sure `transnetv2_infer.py` and
`transnetv2-pytorch-weights.pth` are in the same directory.

---

## Usage

**Process a single video file**

```bash
python transnetv2_infer.py --input /path/to/video.mp4
```

**Process all videos in a directory**

```bash
python transnetv2_infer.py --input /path/to/folder/
```

**Also save visualizations**

```bash
python transnetv2_infer.py --input /path/to/video_or_folder --visualize
```

---

## Output (written next to the input video)

| File                      | Description                                      |
| ------------------------- | ------------------------------------------------ |
| `<video>.predictions.txt` | single‐frame and many‐hot logits                 |
| `<video>.scenes.txt`      | detected scene intervals (start,end)             |
| `<video>.vis.png`         | *(only if `--visualize`)* timeline visualization |
