#!/usr/bin/env python3

import argparse
import flask
from flask_cors import CORS, cross_origin
import io
import time
import threading

import cv2
import numpy as np
from PIL import Image

from modules import core
from modules.face_analyser import get_one_face
import modules.globals
from modules.processors.frame.core import get_frame_processors_modules



parser = argparse.ArgumentParser(description='Deep Fake server')
parser.add_argument('-s', '--source', help='select an source image', dest='source_path')
parser.add_argument('--temporary_image', help='temporary camera image', dest='camera_image_path',
                    default='images/temp.jpg')
parser.add_argument('--device', help='webcam device', dest='device',
                    type=int, default=0)
parser.add_argument('--width', help='width in pixels', dest='width',
                    type=int, default=960)
parser.add_argument('--height', help='height in pixels', dest='height',
                    type=int, default=540)
parser.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory',
                    type=int, default=core.suggest_max_memory())
parser.add_argument('--execution-provider', help='execution provider', dest='execution_provider',
                    default=['coreml'], choices=core.suggest_execution_providers(), nargs='+')
args = parser.parse_args()


def log(msg: str, msg_type: str) -> None:
  print(f"[{msg_type}] {msg}")


def setup() -> None:
  """Set up the face-swapper."""

  modules.globals.source_path = args.source_path
  modules.globals.target_path = None
  modules.globals.output_path = None
  modules.globals.frame_processors = ["face_swapper"]
  modules.globals.headless = None
  modules.globals.keep_fps = False
  modules.globals.keep_audio = False
  modules.globals.keep_frames = False
  modules.globals.many_faces = True
  modules.globals.video_encoder = "libx264"
  modules.globals.video_quality = 18
  modules.globals.max_memory = args.max_memory
  modules.globals.execution_providers = core.decode_execution_providers(args.execution_provider)
  modules.globals.execution_threads = 8
  modules.globals.fp_ui['face_enhancer'] = False
  modules.globals.nsfw = False

  frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
  for frame_processor in frame_processors:
    if not frame_processor.pre_check():
      log("Could not pre-check frame_processor", "error")
      return
  core.limit_resources()


source_image = {"image": None, "annotated_image": None, "timestamp": 0}
current_camera = {"image": None, "byte_string": None, "timestamp": 0}
current_deepfake = {"image": None, "byte_string":None, "timestamp": 0, "active": True}


def copy_from_alt_temp_file() -> None:
  """Capture the source image from the camera."""
  log(f"Copying alt image...", "source")
  cv2_image = cv2.imread("images/deep1.jpg")
  log(f"Image of type {type(cv2_image)}, shape {cv2_image.shape}, max {cv2_image.max()}", "source")
  source_image["image"] = cv2_image
  source_image["annotated_image"] = get_one_face(cv2_image)
  source_image["byte_string"] = write_numpy_to_byte_string(source_image["image"])
  source_image["timestamp"] = time.time()
  log(f"Temporary image saved to {args.camera_image_path}...", "source")
  cv2.imwrite(args.camera_image_path, cv2_image)


def capture_source_image_from_camera() -> None:
  """Capture the source image from the camera."""
  if current_camera["image"] is not None:
    log(f"Capturing camera image...", "source")
    cv2_image = current_camera["image"].copy()
    log(f"Image of type {type(cv2_image)}, shape {cv2_image.shape}, max {cv2_image.max()}", "source")
    source_image["image"] = cv2_image
    source_image["annotated_image"] = get_one_face(cv2_image)
    source_image["byte_string"] = write_numpy_to_byte_string(source_image["image"])
    source_image["timestamp"] = time.time()
    log(f"Temporary image saved to {args.camera_image_path}...", "source")
    cv2.imwrite(args.camera_image_path, cv2_image)


def load_source_image_from_file() -> None:
  """Load the source image from a file."""
  if modules.globals.source_path:
    log(f"Loading image {modules.globals.source_path}...", "source")
    cv2_image = cv2.imread(modules.globals.source_path)
    log(f"Image of type {type(cv2_image)}, shape {cv2_image.shape}, max {cv2_image.max()}", "source")
    source_image["image"] = cv2_image
    source_image["annotated_image"] = get_one_face(cv2_image)
    source_image["byte_string"] = write_numpy_to_byte_string(source_image["image"])
    source_image["timestamp"] = time.time()


def write_numpy_to_byte_string(image: np.ndarray):
  """Write numpy array image onto a bytestream."""
  if image is not None:
    frame = io.BytesIO()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image.save(frame, format='JPEG')
    return frame.getvalue()
  else:
    return None


def write_image_to_byte_string(image: Image.Image):
  """Write PIL image onto a bytestream."""
  if image is not None:
    frame = io.BytesIO()
    image.save(frame, format='JPEG')
    return frame.getvalue()
  else:
    return None


def run_deep_fake_loop() -> None:
  """Run the deep fake loop."""

  # Start the camera.
  cap = cv2.VideoCapture(args.device)  # Use index for the webcam (adjust the index accordingly if necessary)    
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)  # Set the width of the resolution
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)  # Set the height of the resolution
  cap.set(cv2.CAP_PROP_FPS, 60)  # Set the frame rate of the webcam
  PREVIEW_MAX_WIDTH = args.width
  PREVIEW_MAX_HEIGHT = args.height

  # Use the tempoerary face image saved by default.
  load_source_image_from_file()

  # Create the frame processors.
  frame_processors = get_frame_processors_modules(modules.globals.frame_processors)

  # Infinite loop.
  while True:

    # Read the camera and crash if no image.
    camera_return, camera_frame = cap.read()
    if not camera_return:
      log("Cannot get camera input.", "error")
      exit(0)

    # Create a copy of the camera frame and store it.
    current_camera["image"] = camera_frame.copy()
    current_camera["timestamp"] = time.time()
    current_camera["byte_string"] = write_numpy_to_byte_string(current_camera["image"])

    # Process the camera frame to create the deep fake.
    fake_image = camera_frame.copy()
    if current_deepfake["active"] is True:
      try:
        for frame_processor in frame_processors:
          fake_image = frame_processor.process_frame(source_image["annotated_image"], fake_image)
      except:
        log("NEED TO TAKE NEW PICTURE", "error")

    # Convert the image to RGB format to display it with Tkinter and store it.
    current_deepfake["image"] = fake_image
    current_deepfake["byte_string"] = write_numpy_to_byte_string(current_deepfake["image"])
    current_deepfake["timestamp"] = time.time()


thread = threading.Thread(target=run_deep_fake_loop, args=())
thread.start()


def source_stream():
  """Loop that streams the most recent image source."""

  latest_byte_string = None
  latest_timestamp = 0
  while True:
    if latest_timestamp < source_image["timestamp"]:
      log(f"stream: {latest_timestamp}", "source_stream")
      latest_timestamp = source_image["timestamp"]
      latest_byte_string = source_image["byte_string"]
      if latest_byte_string is not None:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_byte_string + b'\r\n')
    else:
      if latest_byte_string:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_byte_string + b'\r\n')
      else:
        time.sleep(0.001)


def deepfake_stream():
  """Loop that streams the camera / deep fake image using the last input / result."""

  latest_byte_string = None
  latest_timestamp = 0
  while True:
    if latest_timestamp < current_deepfake["timestamp"]:
      log(f"stream: {latest_timestamp}", "deepfake_stream")
      latest_timestamp = current_deepfake["timestamp"]
      latest_byte_string = current_deepfake["byte_string"]
      if latest_byte_string is not None:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_byte_string + b'\r\n')
    else:
      if latest_byte_string:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_byte_string + b'\r\n')
      else:
        time.sleep(0.001)


# Define the app, and wrap it in CORS handler.
app = flask.Flask(__name__)
CORS(app, support_credentials=True)


@app.route("/ui")
@cross_origin(supports_credentials=True)
def ui():
  with open("templates/index.html", "r") as f:
    html_ui = f.read()
  return html_ui


@app.route("/copy")
@cross_origin(supports_credentials=True)
def copy():
  copy_from_alt_temp_file()
  return str(source_image["timestamp"])


@app.route("/click")
@cross_origin(supports_credentials=True)
def click():
  capture_source_image_from_camera()
  return str(source_image["timestamp"])


@app.route("/active")
@cross_origin(supports_credentials=True)
def active():
  current_deepfake["active"] = True
  return str("active")


@app.route("/inactive")
@cross_origin(supports_credentials=True)
def inactive():
  current_deepfake["active"] = False
  return str("inactive")


@app.route("/source")
@cross_origin(supports_credentials=True)
def source():
  return flask.Response(source_stream(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/")
@cross_origin(supports_credentials=True)
def stream():
  return flask.Response(deepfake_stream(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
  setup()
  app.run('0.0.0.0', port=8001)
