#!/usr/bin/env python3
"""Process FineVideo Parquet files with YOLO11l on every available NVIDIA GPU.

Example:
    python run.py --input finevideo/data --output results.jsonl
    python run.py --input finevideo/data --output results.jsonl --resume

Completed Parquet files are checkpointed in <output>.parts/. The final JSONL is
merged from those completed parts after all workers finish.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from queue import Empty
from typing import Any

import av
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


MODEL_DEFAULT = "yolo11l.pt"
BATCH_DEFAULT = 16
IMAGE_SIZE = 640
CONFIDENCE = 0.4
NMS_IOU = 0.7
QUANTIZE = 16
WANTED = ("person", "car", "dog", "laptop", "cell phone")


class TimestampError(RuntimeError):
    """Raised only when the first frame has no usable way to establish time."""


def as_fraction(value: Any) -> Fraction | None:
    """Convert a PyAV rational value to Fraction without using float arithmetic."""
    if value is None:
        return None
    try:
        result = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        try:
            result = Fraction(int(value.numerator), int(value.denominator))
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return None
    return result if result > 0 else None


class TimestampTracker:
    """Resolve frame times using PTS, then DTS, then a rational frame interval."""

    def __init__(self, stream: av.video.stream.VideoStream):
        self.stream = stream
        self.average_rate = as_fraction(stream.average_rate)
        self.average_step = (
            Fraction(1, 1) / self.average_rate if self.average_rate else None
        )
        self.first_time: Fraction | None = None
        self.previous_time: Fraction | None = None
        self.last_positive_step: Fraction | None = self.average_step

    def frame_time(self, frame: av.VideoFrame) -> Fraction:
        time_base = as_fraction(frame.time_base) or as_fraction(self.stream.time_base)

        # PTS is presentation time and takes priority. DTS is used only when PTS
        # is unavailable, as requested for damaged or unusual videos.
        absolute_time: Fraction | None = None
        if frame.pts is not None and time_base is not None:
            absolute_time = Fraction(frame.pts) * time_base
        elif frame.dts is not None and time_base is not None:
            absolute_time = Fraction(frame.dts) * time_base

        if absolute_time is None:
            if self.previous_time is None:
                # The first decoded frame defines video time zero. If neither
                # timestamp exists, an average rate is required to establish a
                # valid frame interval; without it, time cannot be derived.
                if self.average_step is None:
                    raise TimestampError(
                        "first frame has no PTS or DTS and stream average_rate is unavailable"
                    )
                absolute_time = Fraction(0, 1)
            else:
                step = self.average_step or self.last_positive_step

                # A frame duration is another exact fallback if the stream rate
                # is unavailable and no positive timestamp interval was seen.
                if step is None and frame.duration and time_base is not None:
                    candidate = Fraction(frame.duration) * time_base
                    if candidate > 0:
                        step = candidate

                # A later frame must not make the entire video fail just because
                # this one frame has no timestamp. If no interval metadata exists
                # at all, keep it at the last known time rather than discarding it.
                if step is None:
                    step = Fraction(0, 1)
                absolute_time = self.previous_time + step

        if self.first_time is None:
            self.first_time = absolute_time

        if self.previous_time is not None:
            delta = absolute_time - self.previous_time
            if delta > 0:
                self.last_positive_step = delta

        self.previous_time = absolute_time
        relative_time = absolute_time - self.first_time

        # Decoded presentation timestamps can be unusual in damaged files. The
        # assignment's timeline begins at the first decoded frame, so do not
        # emit negative second indices if a later timestamp falls before it.
        return max(relative_time, Fraction(0, 1))


def rotate_bgr_frame(image: np.ndarray, frame: av.VideoFrame, stream: Any) -> np.ndarray:
    """Apply the frame's DISPLAYMATRIX rotation and return contiguous BGR data."""
    rotation = 0.0
    try:
        rotation = float(frame.rotation)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        rotation = 0.0

    # Some files expose rotation as a stream metadata tag instead of frame
    # display-matrix side data. Use it only when the frame property is zero.
    if abs(rotation) < 1e-3:
        try:
            rotation = float((stream.metadata or {}).get("rotate", 0))
        except (TypeError, ValueError):
            rotation = 0.0

    if not math.isfinite(rotation):
        rotation = 0.0

    # PyAV defines this angle as counterclockwise. np.rot90 uses the same
    # direction and does exact 90-degree rotations without resampling pixels.
    quarter_turns = round(rotation / 90.0)
    if abs(rotation - quarter_turns * 90.0) < 1e-3:
        if quarter_turns % 4:
            image = np.rot90(image, k=quarter_turns % 4)
    elif abs(rotation) > 1e-3:
        # Display matrices are normally multiples of 90 degrees. Preserve
        # unusual non-right-angle metadata as well, expanding the canvas so the
        # rotated image is not cropped.
        import cv2

        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        cosine = abs(matrix[0, 0])
        sine = abs(matrix[0, 1])
        new_width = int(math.ceil(height * sine + width * cosine))
        new_height = int(math.ceil(height * cosine + width * sine))
        matrix[0, 2] += new_width / 2.0 - center[0]
        matrix[1, 2] += new_height / 2.0 - center[1]
        image = cv2.warpAffine(
            image,
            matrix,
            (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    return np.ascontiguousarray(image)


def summarize_seconds(
    frame_data: dict[int, list[tuple[set[str], int]]],
    video_id: str,
) -> list[dict[str, Any]]:
    """Apply the brief's per-second object-majority and floored-median rules."""
    if not frame_data:
        return []

    output: list[dict[str, Any]] = []
    last_second = max(frame_data)

    # Emit every second from zero to the final processed frame's second. Gaps
    # are represented as empty results, not omitted from the JSONL.
    for second in range(last_second + 1):
        frames = frame_data.get(second, [])
        if not frames:
            output.append(
                {"video": video_id, "second": second, "objects": [], "people": 0}
            )
            continue

        total_frames = len(frames)
        objects = []
        for name in WANTED:
            present_count = sum(1 for found, _ in frames if name in found)
            if present_count * 2 >= total_frames:
                objects.append(name)

        people_counts = sorted(people for _, people in frames)
        count = len(people_counts)
        if count % 2:
            people = people_counts[count // 2]
        else:
            # The brief specifies rounding down when the median is halfway
            # between two integers.
            people = (people_counts[count // 2 - 1] + people_counts[count // 2]) // 2

        output.append(
            {
                "video": video_id,
                "second": second,
                "objects": objects,
                "people": people,
            }
        )

    return output


def _predict_batch(
    batch: list[tuple[int, np.ndarray]],
    model: YOLO,
    gpu_id: int,
) -> dict[int, list[tuple[set[str], int]]]:
    """Run one YOLO batch and return frame-level detections grouped by second."""
    if not batch:
        return {}

    images = [image for _, image in batch]
    results = model(
        images,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=NMS_IOU,
        device=gpu_id,
        verbose=False,
        quantize=QUANTIZE,
    )

    if len(results) != len(batch):
        raise RuntimeError(
            f"model returned {len(results)} results for a batch of {len(batch)} frames"
        )

    batch_data: dict[int, list[tuple[set[str], int]]] = defaultdict(list)
    for (second, _), result in zip(batch, results):
        found: list[str] = []
        if result.boxes is not None and result.boxes.cls is not None:
            for class_id in result.boxes.cls.tolist():
                name = model.names[int(class_id)]
                if name in WANTED:
                    found.append(name)

        batch_data[second].append((set(found), found.count("person")))

    return batch_data


def process_video(
    video_bytes: bytes,
    model: YOLO,
    gpu_id: int,
    batch_size: int,
) -> tuple[dict[int, list[tuple[set[str], int]]], str | None]:
    """Decode one MP4, infer in batches, and retain partial results on errors."""
    frame_data: dict[int, list[tuple[set[str], int]]] = defaultdict(list)
    error_message: str | None = None
    container = None

    try:
        container = av.open(io.BytesIO(video_bytes), mode="r")
        if not container.streams.video:
            raise RuntimeError("video stream not found")

        stream = container.streams.video[0]
        timestamps = TimestampTracker(stream)
        batch: list[tuple[int, np.ndarray]] = []

        def run_and_store(current_batch: list[tuple[int, np.ndarray]]) -> None:
            if not current_batch:
                return
            detected = _predict_batch(current_batch, model, gpu_id)
            for second, frame_results in detected.items():
                frame_data[second].extend(frame_results)

        try:
            for frame in container.decode(stream):
                relative_time = timestamps.frame_time(frame)
                second = relative_time.numerator // relative_time.denominator

                # YOLO's input is BGR. PyAV converts directly to BGR24 here.
                image = frame.to_ndarray(format="bgr24")
                image = rotate_bgr_frame(image, frame, stream)
                batch.append((second, image))

                if len(batch) >= batch_size:
                    try:
                        run_and_store(batch)
                    except Exception as exc:
                        error_message = f"inference failed: {type(exc).__name__}: {exc}"
                        batch = []
                        break
                    batch = []

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

        # Process decoded frames still waiting in the final partial batch, even
        # when the decoder stopped partway through the video.
        if batch:
            try:
                run_and_store(batch)
            except Exception as exc:
                batch_error = f"inference failed: {type(exc).__name__}: {exc}"
                error_message = (
                    f"{error_message}; {batch_error}"
                    if error_message
                    else batch_error
                )

        if not frame_data and error_message is None:
            error_message = "no frames were successfully processed"

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"

    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass

    return frame_data, error_message


def jsonl_write(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
    handle.write("\n")


def write_file_error_part(parquet_path: Path, part_path: Path, message: str) -> None:
    """Create an error-only part if a Parquet file or worker cannot be processed."""
    tmp_path = Path(str(part_path) + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        # A Parquet-level failure can occur before row metadata is readable.
        # Row zero is used as a file-level error marker in that rare case.
        jsonl_write(
            handle,
            {"video": f"{parquet_path.name}:0", "error": message[:500]},
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, part_path)


def process_parquet_file(
    parquet_path: Path,
    parts_dir: Path,
    model: YOLO,
    gpu_id: int,
    batch_size: int,
) -> None:
    """Process all video rows in one Parquet file into an atomic part file."""
    part_path = parts_dir / f"{parquet_path.name}.jsonl"
    tmp_path = Path(str(part_path) + ".tmp")
    started = time.perf_counter()
    print(f"[GPU {gpu_id}] Starting {parquet_path.name}", flush=True)

    try:
        dataframe = pd.read_parquet(parquet_path, columns=["mp4"])
    except Exception as exc:
        write_file_error_part(
            parquet_path,
            part_path,
            f"could not read Parquet file: {type(exc).__name__}: {exc}",
        )
        print(f"[GPU {gpu_id}] Parquet read failed: {parquet_path.name}: {exc}", flush=True)
        return

    try:
        with tmp_path.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
            for row_number, video_value in enumerate(dataframe["mp4"]):
                video_id = f"{parquet_path.name}:{row_number}"
                try:
                    if video_value is None or (isinstance(video_value, float) and math.isnan(video_value)):
                        raise ValueError("empty mp4 column value")

                    video_bytes = bytes(video_value)
                    frame_data, video_error = process_video(
                        video_bytes, model, gpu_id, batch_size
                    )

                    for record in summarize_seconds(frame_data, video_id):
                        jsonl_write(handle, record)

                    if video_error is not None:
                        jsonl_write(
                            handle,
                            {"video": video_id, "error": video_error[:500]},
                        )

                except Exception as exc:
                    # Keep processing later rows even if this row fails.
                    jsonl_write(
                        handle,
                        {
                            "video": video_id,
                            "error": f"{type(exc).__name__}: {exc}"[:500],
                        },
                    )

            handle.flush()
            os.fsync(handle.fileno())

        # A .jsonl part exists only after every row in this Parquet file has
        # been attempted. A leftover .tmp is never treated as complete.
        os.replace(tmp_path, part_path)
        elapsed = time.perf_counter() - started
        print(
            f"[GPU {gpu_id}] Finished {parquet_path.name} "
            f"({len(dataframe)} videos) in {elapsed:.1f}s",
            flush=True,
        )

    except Exception:
        # Do not promote an incomplete temporary file to a completed checkpoint.
        print(
            f"[GPU {gpu_id}] Failed while writing {parquet_path.name}:\n"
            f"{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )
        raise


def gpu_worker(
    gpu_id: int,
    task_queue: Any,
    parts_dir: str,
    model_path: str,
    batch_size: int,
    worker_count: int,
) -> None:
    """One long-lived process and one YOLO model per GPU."""
    try:
        torch.cuda.set_device(gpu_id)
        cpu_count = os.cpu_count() or worker_count
        torch.set_num_threads(max(1, cpu_count // max(1, worker_count)))

        model = YOLO(model_path)

        # Warm up with the same batch size and inference precision used below.
        dummy_batch = [
            np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]
        model(
            dummy_batch,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            device=gpu_id,
            verbose=False,
            quantize=QUANTIZE,
        )
        del dummy_batch
        print(f"[GPU {gpu_id}] Model ready", flush=True)

    except Exception:
        print(
            f"[GPU {gpu_id}] Model initialization failed:\n{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )
        return

    parts_path = Path(parts_dir)
    while True:
        try:
            parquet_path_string = task_queue.get(timeout=1)
        except Empty:
            continue

        if parquet_path_string is None:
            break

        parquet_path = Path(parquet_path_string)
        try:
            process_parquet_file(
                parquet_path,
                parts_path,
                model,
                gpu_id,
                batch_size,
            )
        except Exception as exc:
            print(
                f"[GPU {gpu_id}] Worker error on {parquet_path.name}: {exc}",
                file=sys.stderr,
                flush=True,
            )


def run_workers(
    files: list[Path],
    gpu_ids: list[int],
    parts_dir: Path,
    model_path: str,
    batch_size: int,
) -> None:
    """Distribute Parquet files dynamically among one process per GPU."""
    if not files:
        return

    worker_count = min(len(gpu_ids), len(files))
    active_gpu_ids = gpu_ids[:worker_count]
    context = mp.get_context("spawn")
    task_queue = context.Queue()

    for path in files:
        task_queue.put(str(path))
    for _ in active_gpu_ids:
        task_queue.put(None)

    processes = [
        context.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                task_queue,
                str(parts_dir),
                model_path,
                batch_size,
                worker_count,
            ),
            name=f"yolo-gpu-{gpu_id}",
        )
        for gpu_id in active_gpu_ids
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    task_queue.close()
    task_queue.join_thread()


def merge_parts(files: list[Path], parts_dir: Path, output_path: Path) -> None:
    """Merge completed per-Parquet JSONL files in input-file order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_tmp = Path(str(output_path) + ".tmp")
    missing = []

    with merged_tmp.open("w", encoding="utf-8", buffering=1024 * 1024) as output:
        for parquet_path in files:
            part_path = parts_dir / f"{parquet_path.name}.jsonl"
            if not part_path.exists():
                missing.append(parquet_path)
                continue
            with part_path.open("r", encoding="utf-8") as part:
                shutil.copyfileobj(part, output)
        output.flush()
        os.fsync(output.fileno())

    os.replace(merged_tmp, output_path)

    if missing:
        print(
            f"Warning: {len(missing)} Parquet file(s) did not produce a part file.",
            file=sys.stderr,
        )
        for path in missing:
            print(f"  {path.name}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO11l video indexing on FineVideo Parquet files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing .parquet files (the files themselves, not their parent directory).",
    )
    parser.add_argument("--output", required=True, help="Final JSONL output path.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip Parquet files that already have completed part files.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        help="Path to the official YOLO11l weights file (default: yolo11l.pt).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_DEFAULT,
        help=f"Frames per YOLO inference batch (default: {BATCH_DEFAULT}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    parts_dir = Path(str(output_path) + ".parts")
    model_path = str(Path(args.model).expanduser())

    if args.batch_size < 1:
        print("Error: --batch-size must be at least 1.", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"Error: input folder does not exist: {input_dir}", file=sys.stderr)
        return 2
    if not Path(model_path).is_file():
        print(
            f"Error: model weights not found: {model_path}\n"
            "Download yolo11l.pt during the preparation step before running offline.",
            file=sys.stderr,
        )
        return 2

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"Error: no .parquet files found directly inside {input_dir}", file=sys.stderr)
        return 2

    try:
        gpu_count = torch.cuda.device_count()
    except Exception as exc:
        print(f"Error: could not inspect CUDA devices: {exc}", file=sys.stderr)
        return 2

    if gpu_count < 1:
        print("Error: no NVIDIA CUDA GPU is available. This task requires a GPU.", file=sys.stderr)
        return 2

    parts_dir.mkdir(parents=True, exist_ok=True)

    if not args.resume:
        # A normal run starts clean: old completed parts and stale .tmp files
        # are removed so every Parquet file is processed again.
        shutil.rmtree(parts_dir)
        parts_dir.mkdir(parents=True, exist_ok=True)

    completed = {
        path.name[: -len(".jsonl")]
        for path in parts_dir.glob("*.jsonl")
        if path.is_file()
    }
    pending = [path for path in parquet_files if path.name not in completed]

    print(f"Parquet files found: {len(parquet_files)}")
    print(f"Completed checkpoints found: {len(completed)}")
    print(f"Files to process: {len(pending)}")
    print(f"CUDA GPUs detected: {gpu_count}")
    print(f"Inference batch size: {args.batch_size}")
    print(f"Resume mode: {args.resume}")

    if pending:
        run_workers(
            pending,
            list(range(gpu_count)),
            parts_dir,
            model_path,
            args.batch_size,
        )

        # If a worker died unexpectedly, retry any file that has no completed
        # part. This also handles a worker that failed during model startup.
        missing = [
            path
            for path in pending
            if not (parts_dir / f"{path.name}.jsonl").exists()
        ]
        if missing:
            print(
                f"Retrying {len(missing)} unfinished Parquet file(s) on GPU 0.",
                flush=True,
            )
            run_workers(
                missing,
                [0],
                parts_dir,
                model_path,
                args.batch_size,
            )

        # Do not silently omit an input file if it still cannot be started.
        for path in missing:
            part_path = parts_dir / f"{path.name}.jsonl"
            if not part_path.exists():
                write_file_error_part(
                    path,
                    part_path,
                    "worker could not complete this Parquet file; inspect stderr",
                )

    merge_parts(parquet_files, parts_dir, output_path)
    print(f"Wrote final results: {output_path}")
    print(f"Per-Parquet checkpoints: {parts_dir}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
