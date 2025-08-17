# TransNetV2 PyTorch Inference

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

Install this tool as a package:
```sh
pip install -e .
```

Make sure `inference.py` and
`transnetv2-pytorch-weights.pth` are in the same directory.

---

## CLI Usage

**Process a single video file**

```bash
transnetv2_pytorch /path/to/video.mp4
```

**Process all videos in a directory**

```bash
transnetv2_pytorch /path/to/folder/
```

**Also save visualizations**

```bash
transnetv2_pytorch /path/to/video_or_folder --visualize
```

---

## Docker Usage
Build the container:
```sh
docker build -t transnetv2_pytorch .
```

Run on a directory of videos (GPU):
```sh
docker run --rm --gpus all \
  -v /my/local/videos:/data \
  transnetv2_pytorch /data
```

Run on a single video (CPU or GPU):
```sh
docker run --rm --gpus all \
  -v /my/local/videos:/data \
  transnetv2_pytorch /data/video1.mp4
```
## Output (written next to the input video)

| File                      | Description                                      |
| ------------------------- | ------------------------------------------------ |
| `<video>.predictions.txt` | single‐frame and many‐hot logits                 |
| `<video>.scenes.txt`      | detected scene intervals (start,end)             |
| `<video>.vis.png`         | *(only if `--visualize`)* timeline visualization |
