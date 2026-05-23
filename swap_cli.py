#!/usr/bin/env python3

import argparse
import sys
import os

# Add root directory to python path if not already present
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_deep_fake import FaceSwapper
from modules import core
import deep_fake

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.gif'}


def main():
  parser = argparse.ArgumentParser(
      description="Unified CLI tool to swap faces in an image or video using Deep-Live-Cam FaceSwapper."
  )
  parser.add_argument('-s', '--source', help='Path to source face image (e.g. images/temp.jpg)', required=True)
  parser.add_argument('-t', '--target', help='Path to target image or video (e.g. images/target.jpg or video.mp4)', required=True)
  parser.add_argument('-o', '--output', default=None, help='Path to save processed file (defaults to images/output.jpg for images or images/output.mp4 for videos)')
  parser.add_argument('--execution-provider', help='ONNX execution provider (e.g. cpu, cuda, directml)', default='cuda')
  parser.add_argument('--max-memory', type=int, default=None, help='Maximum amount of RAM in GB')
  args = parser.parse_args()

  # Check if target exists
  if not os.path.exists(args.target):
    print(f"Error: Target file '{args.target}' does not exist.", file=sys.stderr)
    sys.exit(1)

  # Determine target type by file extension
  ext = os.path.splitext(args.target)[1].lower()
  is_image = ext in IMAGE_EXTENSIONS
  is_video = ext in VIDEO_EXTENSIONS
  if not is_image and not is_video:
    print(f"Error: Unsupported target file format '{ext}'. Must be an image or a video.", file=sys.stderr)
    sys.exit(1)

  # Assign default output path if not explicitly provided
  output_path = args.output
  if not output_path:
    if is_image:
      output_path = 'images/output.jpg'
    else:
      output_path = 'images/output.mp4'

  try:
    if is_image:
      deep_fake.swap_image(
          source_path=args.source,
          target_path=args.target,
          output_path=output_path,
          execution_provider=args.execution_provider,
          max_memory=args.max_memory,
          verbose=True
      )
    else:
      deep_fake.swap_video(
          source_path=args.source,
          target_path=args.target,
          output_path=output_path,
          execution_provider=args.execution_provider,
          max_memory=args.max_memory,
          verbose=True
      )
  except KeyboardInterrupt:
    print("\nAborted by user.")
    sys.exit(1)
  except Exception as e:
    print(f"\nError: {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
