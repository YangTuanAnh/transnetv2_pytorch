#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
import ffmpeg
from tqdm import tqdm

from transnetv2_pytorch import TransNetV2


class TransNetV2Torch:
    def __init__(self):
        self._input_size = (27, 48, 3)
        # assume the weights are located in the same directory as this script
        weights_path = os.path.join(os.path.dirname(__file__),
                                    'transnetv2-pytorch-weights.pth')
        self.model = TransNetV2()
        self.model.load_state_dict(torch.load(weights_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.eval().to(self.device)
        self.window_buf = None

    @torch.no_grad()
    def predict_raw(self, frames_tensor):
        logits, out = self.model(frames_tensor)
        return logits.cpu().numpy(), out["many_hot"].cpu().numpy()

    def predict_frames(self, frames: np.ndarray):
        total = len(frames)
        pad_start = np.repeat(frames[0:1], 25, axis=0)
        pad_end_len = 25 + (50 - (total % 50) if total % 50 != 0 else 50)
        pad_end = np.repeat(frames[-1:], pad_end_len, axis=0)
        padded = np.concatenate((pad_start, frames, pad_end), axis=0)

        if self.window_buf is None:
            self.window_buf = np.empty((1, 100, *self._input_size),
                                       dtype=np.uint8)

        logits_list, manyhot_list = [], []
        ptr = 0
        while ptr + 100 <= len(padded):
            self.window_buf[0] = padded[ptr:ptr + 100]
            tensor_input = torch.from_numpy(self.window_buf).to(self.device)
            l, mh = self.predict_raw(tensor_input)
            logits_list.append(l[0, 25:75, 0])
            manyhot_list.append(mh[0, 25:75, 0])

            ptr += 50
            processed = min(ptr, total)
            print(f"\r[TransNetV2] Processing video frames {processed}/{total}", end="")

        print("")  # newline after bar

        logits = np.concatenate(logits_list, axis=0)[:total]
        manyhot = np.concatenate(manyhot_list, axis=0)[:total]

        # sigmoid in numpy
        s_pred = 1.0 / (1.0 + np.exp(-logits))
        a_pred = 1.0 / (1.0 + np.exp(-manyhot))
        return s_pred, a_pred

    def predict_video(self, video_path: str):
        print(f"[TransNetV2] Extracting frames from {os.path.basename(video_path)}")
        video_stream, _ = ffmpeg.input(video_path).output(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
        ).run(capture_stdout=True, capture_stderr=True)

        frames = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
        return frames, *self.predict_frames(frames)

    @staticmethod
    def predictions_to_scenes(preds, threshold=0.5):
        preds = (preds > threshold).astype(np.uint8)
        scenes, t_prev, start = [], 0, 0
        for i, t in enumerate(preds):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])
        if not scenes:
            return np.array([[0, len(preds) - 1]], dtype=np.int32)
        return np.array(scenes, dtype=np.int32)

    @staticmethod
    def visualize_predictions(frames: np.ndarray, predictions):
        from PIL import Image, ImageDraw

        if isinstance(predictions, np.ndarray):
            predictions = [predictions]

        ih, iw, _ = frames.shape[1:]
        width = 25
        pad_with = width - len(frames) % width if len(frames) % width != 0 else 0

        frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)), (0, 0)])
        predictions = [np.pad(x, (0, pad_with)) for x in predictions]
        height = len(frames) // width

        img = frames.reshape([height, width, ih + 1, iw + len(predictions), 3])
        img = np.concatenate(np.split(
            np.concatenate(np.split(img, height), axis=2)[0], width
        ), axis=2)[0, :-1]

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        for i, pred in enumerate(zip(*predictions)):
            x, y = i % width, i // width
            x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1
            for j, p in enumerate(pred):
                color = [0, 0, 0]
                color[(j + 1) % 3] = 255
                value = round(p * (ih - 1))
                if value != 0:
                    draw.line((x + j, y, x + j, y - value),
                              fill=tuple(color), width=1)
        return img


# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                    help="video file or directory containing videos")
    parser.add_argument("--visualize", action="store_true",
                        help="also save <video>.vis.png")
    args = parser.parse_args()

    model = TransNetV2Torch()

    # get list of files
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f)
                 for f in os.listdir(args.input)]
    else:
        files = [args.input]

    pbar = tqdm(files)
    for fp in pbar:
        pbar.set_description(os.path.basename(fp))

        pred_txt = fp + ".predictions.txt"
        scenes_txt = fp + ".scenes.txt"

        if os.path.exists(pred_txt) or os.path.exists(scenes_txt):
            print(f"[SKIP] {fp} already processed. Skipping.", file=sys.stderr)
            continue

        frames, s_pred, a_pred = model.predict_video(fp)

        pred_arr = np.stack([s_pred, a_pred], axis=1)
        np.savetxt(pred_txt, pred_arr, fmt="%.6f")
        scenes = model.predictions_to_scenes(s_pred)
        np.savetxt(scenes_txt, scenes, fmt="%d")

        if args.visualize:
            vis_img = model.visualize_predictions(
                frames, predictions=(s_pred, a_pred))
            vis_img.save(fp + ".vis.png")


if __name__ == "__main__":
    main()
