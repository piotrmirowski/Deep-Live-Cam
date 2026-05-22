#!/usr/bin/env python3

import argparse
import sys
import os
import time
import cv2

# Add root directory to python path if not already present
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_deep_fake import FaceSwapper
from modules import core
import modules.globals

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=40, fill='#', print_end="\r"):
    """
    Call in a loop to create terminal progress bar.
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total:
        print()

def merge_audio(temp_video_path, original_video_path, output_video_path):
    """
    Merge the original audio from the target video into the face-swapped temporary video.
    """
    import subprocess
    import shutil
    
    print("--------------------------------------------------")
    print("Merging original audio into final video...")
    
    # ffmpeg -y -i temp_video -i original_video -c:v copy -c:a copy -map 0:v:0 -map 1:a? output_video
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", temp_video_path,
        "-i", original_video_path,
        "-c:v", "copy",
        "-c:a", "copy",
        "-map", "0:v:0",
        "-map", "1:a?",
        output_video_path
    ]
    
    try:
        subprocess.run(command, check=True)
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            print("Successfully merged audio!")
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return True
    except Exception as e:
        print(f"Warning: Failed to merge audio using FFmpeg: {e}")
        print("Falling back to silent video.")
        if os.path.exists(output_video_path):
            try:
                os.remove(output_video_path)
            except Exception:
                pass
        shutil.move(temp_video_path, output_video_path)
        return False

def main():
    parser = argparse.ArgumentParser(description="CLI tool to swap faces in a video using Deep-Live-Cam FaceSwapper.")
    parser.add_argument('-s', '--source', help='Path to source face image (e.g. images/temp.jpg)', required=True)
    parser.add_argument('-t', '--target', help='Path to target video (e.g. images/Man_speaks_to_camera_202605221602.mp4)', required=True)
    parser.add_argument('-o', '--output', default='images/output.mp4', help='Path to save processed video (e.g. images/output.mp4)')
    parser.add_argument('--execution-provider', help='ONNX execution provider (e.g. cpu, cuda, directml)', default='cuda')
    parser.add_argument('--max-memory', type=int, default=core.suggest_max_memory(), help='Maximum amount of RAM in GB')

    args = parser.parse_args()

    # Verify files exist
    if not os.path.exists(args.source):
        print(f"Error: Source image file '{args.source}' does not exist.")
        sys.exit(1)
    if not os.path.exists(args.target):
        print(f"Error: Target video file '{args.target}' does not exist.")
        sys.exit(1)

    # Make sure output directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Mock opts object expected by FaceSwapper
    class FaceSwapperOpts:
        def __init__(self):
            self.source_path = args.source
            self.width = 960
            self.height = 540
            self.max_memory = args.max_memory
            self.execution_provider = [args.execution_provider]
            self.cli_mode = True

    opts = FaceSwapperOpts()

    print("--------------------------------------------------")
    print("Initializing FaceSwapper in CLI mode...")
    print(f"Source Image: {args.source}")
    print(f"Target Video: {args.target}")
    print(f"Output Video: {args.output}")
    print(f"Execution Provider: {args.execution_provider}")
    print("--------------------------------------------------")

    swapper = FaceSwapper(opts)

    # Enable "many_faces" to swap all faces in the frame
    swapper.many_faces(True)

    # Get the annotated source face
    source_face = swapper.source_image["annotated_image"]
    if not source_face:
        print(f"Error: No face detected in the source image '{args.source}'.")
        sys.exit(1)

    print("Source face loaded successfully.")

    # Open target video
    cap = cv2.VideoCapture(args.target)
    if not cap.isOpened():
        print(f"Error: Could not open target video '{args.target}' using OpenCV.")
        sys.exit(1)

    # Get video metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        total_frames = 1  # prevent division by zero in progress bar if frames count is missing

    print(f"Target Video Resolution: {width}x{height}")
    print(f"Target Video FPS: {fps:.2f}")
    print(f"Total Frames to Process: {total_frames}")
    print("--------------------------------------------------")

    # Initialize video writer
    temp_output = args.output + ".temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open output video writer for '{args.output}'.")
        cap.release()
        sys.exit(1)

    start_time = time.time()
    processed_count = 0

    try:
        print_progress_bar(0, total_frames, prefix='Processing:', suffix='Complete', length=40)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Swap faces in the frame
            # swapper._process_frame takes source face and frame, and returns processed frame
            processed_frame = swapper._process_frame(source_face, frame)

            # Write the processed frame to output video file
            out.write(processed_frame)

            processed_count += 1
            elapsed = time.time() - start_time
            fps_speed = processed_count / elapsed if elapsed > 0 else 0
            suffix = f'{processed_count}/{total_frames} ({fps_speed:.1f} fps)'
            print_progress_bar(processed_count, total_frames, prefix='Processing:', suffix=suffix, length=40)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during face swapping: {e}")
    finally:
        cap.release()
        out.release()

    total_time = time.time() - start_time
    print("--------------------------------------------------")
    print(f"Face swapping completed in {total_time:.2f} seconds.")
    
    # Merge original audio from the target video into the face-swapped video
    merge_audio(temp_output, args.target, args.output)

    if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        print(f"Successfully saved output video to: {args.output}")
        print(f"Output File Size: {os.path.getsize(args.output) / (1024 * 1024):.2f} MB")
    else:
        print("Error: Output video file is empty or was not created.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
