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
import deep_fake


def main():
  parser = argparse.ArgumentParser(description="CLI tool to swap faces in a video using Deep-Live-Cam FaceSwapper.")
  parser.add_argument('-s', '--source', help='Path to source face image (e.g. images/temp.jpg)', required=True)
  parser.add_argument('-t', '--target', help='Path to target video (e.g. images/Man_speaks_to_camera_202605221602.mp4)', required=True)
  parser.add_argument('-o', '--output', default='images/output.mp4', help='Path to save processed video (e.g. images/output.mp4)')
  parser.add_argument('--execution-provider', help='ONNX execution provider (e.g. cpu, cuda, directml)', default='cuda')
  parser.add_argument('--max-memory', type=int, default=None, help='Maximum amount of RAM in GB')

  args = parser.parse_args()

  try:
    deep_fake.swap_video(
      source_path=args.source,
      target_path=args.target,
      output_path=args.output,
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
