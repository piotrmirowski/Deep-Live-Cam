#!/usr/bin/env python3

import argparse
import flask
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO
import logging
import os
import requests
import sys
import time
from ddgs import ddgs
import threading
import cv2
import numpy as np
import rembg
import pyvirtualcam
import colorsys

from modules import core
from modules.face_analyser import get_one_face, get_many_faces
import modules.globals
from modules.processors.frame.core import get_frame_processors_modules
from modules.processors.frame.face_swapper import swap_face
from modules.typing import Face, Frame
import utils


_TEMPORARY_IMAGE_PATH = "images/temp.jpg"
_CAMERA_IMAGE_PATH = "images/camera.jpg"
_KEYS = ['bbox', 'kps', 'gender', 'age']


parser = argparse.ArgumentParser(description='Deep Fake server')
parser.add_argument('-s', '--source', help='select an source image', dest='source_path',
                    default="templates/einstein.jpg")
parser.add_argument('--port', help='Port', dest='port', type=int, default=8001)
parser.add_argument('--device', help='webcam device', dest='device',
                    type=str, default="Integrated Webcam")
parser.add_argument('--camera-index', help='webcam device index', dest='camera_index',
                    type=int, default=None)
parser.add_argument('--width', help='width in pixels', dest='width',
                    type=int, default=960)
parser.add_argument('--height', help='height in pixels', dest='height',
                    type=int, default=540)
parser.add_argument('--num-search-images', help='number of images to search by ddgs', dest='num_search_images',
                    type=int, default=3)
parser.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory',
                    type=int, default=core.suggest_max_memory())
parser.add_argument('--execution-provider', help='execution provider', dest='execution_provider',
                    default=['coreml'], choices=core.suggest_execution_providers(), nargs='+')
opts = parser.parse_args()


def log(msg: str, msg_type: str) -> None:
  print(f"[{msg_type}] {msg}")


class FaceSwapper(object):

  def __init__(self, opts):
    # Make sure the images directory exists
    if not os.path.exists("images"):
      os.makedirs("images")

    # Initialise the parameters
    self._source_path = opts.source_path
    if getattr(opts, 'camera_index', None) is not None:
      utils.list_webcams(opts.device)
      self._device = opts.camera_index
      log(f"Using camera index {self._device} specified by --camera-index", "info")
    else:
      self._device = utils.list_webcams(opts.device)
    self._width = opts.width
    self._height = opts.height
    self._init(opts)

    # Current image and deepfake storage
    self.source_image = {"image": None, "annotated_image": None, "timestamp": 0}
    self.current_camera = {"image": None, "byte_string": None, "timestamp": 0}
    self.current_deepfake = {"image": None, "byte_string":None, "timestamp": 0, "active": False}
    self.current_faces = []
    self.target_embedding = None

    # Start the camera.
    self._cap = cv2.VideoCapture(self._device)  # Use index for the webcam (adjust the index accordingly if necessary)    
    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)  # Set the width of the resolution
    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)  # Set the height of the resolution
    self._cap.set(cv2.CAP_PROP_FPS, 60)  # Set the frame rate of the webcam

    # Set up the frame processors
    self.setup()

    # Use the tempoerary face image saved by default.
    self._load_source_image_from_file()

    # Initialize the rembg session
    self.rembg_session = rembg.new_session()
    self.background_removal = False

    # Start the deep fake processing
    self._thread = None
    self.start()


  def _init(self, opts):
    modules.globals.source_path = self._source_path
    modules.globals.target_path = None
    modules.globals.output_path = None
    modules.globals.frame_processors = ["face_swapper"]
    modules.globals.headless = None
    modules.globals.keep_fps = False
    modules.globals.keep_audio = False
    modules.globals.keep_frames = False
    modules.globals.many_faces = False
    modules.globals.video_encoder = "libx264"
    modules.globals.video_quality = 18
    modules.globals.max_memory = opts.max_memory
    modules.globals.execution_providers = core.decode_execution_providers(opts.execution_provider)
    modules.globals.execution_threads = 8
    modules.globals.fp_ui['face_enhancer'] = False
    modules.globals.nsfw = False


  def reset_target_embedding(self):
    self.target_embedding = None


  def background_removal(self, value: bool):
    self.background_removal = value


  def many_faces(self, value: bool):
    prev_value = modules.globals.many_faces
    modules.globals.many_faces = value
    if value != prev_value:
      self.setup()


  def setup(self) -> None:
    """Set up the face-swapper."""

    self._frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
    for frame_processor in self._frame_processors:
      if not frame_processor.pre_check():
        log("Could not pre-check frame_processor", "error")
        exit(1)
    core.limit_resources()


  def status(self):
    return {"many_faces": modules.globals.many_faces,
            "faces": self.current_faces,
            "active": self.current_deepfake["active"],
            "background_removal": self.background_removal}


  def _store_source_image(self, cv2_image):
    if cv2_image is not None:
      log(f"Image of type {type(cv2_image)}, shape {cv2_image.shape}, max {cv2_image.max()}", "source")
      self.source_image["image"] = cv2_image
      self.source_image["annotated_image"] = get_one_face(cv2_image)
      self.source_image["byte_string"] = utils.write_numpy_to_byte_string(self.source_image["image"])
      self.source_image["timestamp"] = time.time()


  def read_source_image_from_file(self) -> None:
    """Read the source image from a file."""
    log(f"Reading image from {_TEMPORARY_IMAGE_PATH}...", "source")
    cv2_image = cv2.imread(_TEMPORARY_IMAGE_PATH)
    self._store_source_image(cv2_image)


  def capture_source_image_from_camera(self) -> None:
    """Capture the source image from the camera."""
    if self.current_camera["image"] is not None:
      log(f"Capturing camera image, storing in {_CAMERA_IMAGE_PATH}...", "source")
      cv2_image = self.current_camera["image"].copy()
      cv2.imwrite(_CAMERA_IMAGE_PATH, cv2_image)
      cv2.imwrite(_TEMPORARY_IMAGE_PATH, cv2_image)
      self._store_source_image(cv2_image)


  def _load_source_image_from_file(self) -> None:
    """Load the source image from a file."""
    if self._source_path:
      log(f"Loading image {self._source_path}...", "source")
      cv2_image = cv2.imread(self._source_path)
      cv2.imwrite(_CAMERA_IMAGE_PATH, cv2_image)
      cv2.imwrite(_TEMPORARY_IMAGE_PATH, cv2_image)
      self._store_source_image(cv2_image)


  def _store_face_stats(self, faces):
    """Store the face stats."""
    new_face_stats = []
    if faces:
      for face in faces:
        face_stats = {}
        for key in _KEYS:
          val = face[key]
          if hasattr(val, 'tolist'):
            val = val.tolist()
          elif hasattr(val, 'item'):
            val = val.item()
          face_stats[key] = val
        new_face_stats.append(face_stats)
    self.current_faces = new_face_stats


  def _process_frame(self, source_face: Face, temp_frame: Frame) -> Frame:
    """Reimplementation of process_frame from modules/processors/frame/face_swapper.py
    but with possibility to track one specific face in the output frame."""
    if modules.globals.color_correction:
      temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)

    # Early exit if no source face is given.
    if not source_face:
      log("No source face found.", "error")
      return temp_frame

    # Get all the faces in the temp frame.
    many_faces = get_many_faces(temp_frame)
    self._store_face_stats(many_faces)


    # Early exit if no target face is found.
    if not many_faces:
      log("No target faces found.", "error")
      return temp_frame

    if modules.globals.many_faces:
      for target_face in many_faces:
        temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
      if self.target_embedding is None:
        self.target_embedding = many_faces[0].normed_embedding
        target_face = many_faces[0]
      else:
        best_face = many_faces[0]
        best_sim = -1
        for face in many_faces:
          emb1 = self.target_embedding
          emb2 = face.normed_embedding
          similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
          if similarity > best_sim:
            best_sim = similarity
            best_face = face
        target_face = best_face
      temp_frame = swap_face(source_face, target_face, temp_frame)

    return temp_frame


  def _run_deep_fake_loop(self) -> None:
    """Run the deep fake loop."""

    t_frame = time.time()
    delta_t_frame = 0

    # for _ in range(4):
    #   # Read the camera and crash if no image.
    #   camera_return, camera_frame = self._cap.read()
    #   if not camera_return:
    #     log("Cannot get camera input.", "error")
    #     exit(0)
    #   time.sleep(0.033)
    #   delta_t_frame = 0.5 * (time.time() - t_frame) + delta_t_frame * 0.5
    #   t_frame = time.time()
    #   log(f"FPS: {1/delta_t_frame:.2f}", "info")

    while True:

      # Read the camera and crash if no image.
      camera_return, camera_buffer = self._cap.read()
      if not camera_return:
        log("Cannot get camera input.", "error")
        exit(0)
      camera_frame = camera_buffer.copy()

      # Resize the camera frame to the specified width and height.
      t0 = time.time()
      camera_frame = cv2.resize(camera_frame, (self._width, self._height))
      delta_t_size = time.time() - t0

      # Create a copy of the camera frame and store it.
      self.current_camera["image"] = camera_frame.copy()
      self.current_camera["timestamp"] = time.time()
      self.current_camera["byte_string"] = utils.write_numpy_to_byte_string(self.current_camera["image"])

      # Process the camera frame to create the deep fake.
      fake_image = camera_frame.copy()
      delta_t_bg = 0
      if self.background_removal:
        t0 = time.time()
        fake_image = rembg.remove(fake_image, session=self.rembg_session)[:][:, :, :3]
        delta_t_bg = time.time() - t0
      delta_t_swap = 0
      if self.current_deepfake["active"] is True:
        t0 = time.time()
        source_face = self.source_image["annotated_image"]
        fake_image = self._process_frame(source_face, fake_image)
        delta_t_swap = time.time() - t0
      else:
        self.target_embedding = None
        many_faces = get_many_faces(fake_image)
        self._store_face_stats(many_faces)

      # Convert the image to RGB format to display it with Tkinter and store it.
      self.current_deepfake["image"] = fake_image
      self.current_deepfake["byte_string"] = utils.write_numpy_to_byte_string(self.current_deepfake["image"])
      self.current_deepfake["timestamp"] = time.time()

      with pyvirtualcam.Camera(width=self._width, height=self._height, fps=20) as cam:
        cam.send(cv2.cvtColor(fake_image, cv2.COLOR_BGR2RGB))
      delta_t_frame = 0.5 * (time.time() - t_frame) + delta_t_frame * 0.5
      t_frame = time.time()
      log(f"FPS: {1/delta_t_frame:.2f}, size: {delta_t_size:.2f}s, bg: {delta_t_bg:.2f}s, swap: {delta_t_swap:.2f}s", "info")


  def start(self):
    self._thread = threading.Thread(target=self._run_deep_fake_loop, args=(), daemon=True)
    self._thread.start()



def source_stream(face_swapper: FaceSwapper):
  """Loop that streams the most recent image source."""

  latest_byte_string = None
  latest_timestamp = 0
  while True:
    if latest_timestamp < face_swapper.source_image["timestamp"]:
      log(f"stream: {latest_timestamp}", "source_stream")
      latest_timestamp = face_swapper.source_image["timestamp"]
      latest_byte_string = face_swapper.source_image["byte_string"]
      if latest_byte_string is not None:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_byte_string + b'\r\n')
    else:
      if latest_byte_string:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_byte_string + b'\r\n')
      else:
        time.sleep(0.001)


def deepfake_stream(face_swapper: FaceSwapper):
  """Loop that streams the camera / deep fake image using the last input / result."""

  latest_byte_string = None
  latest_timestamp = 0
  while True:
    if latest_timestamp < face_swapper.current_deepfake["timestamp"]:
      log(f"stream: {latest_timestamp}", "deepfake_stream")
      latest_timestamp = face_swapper.current_deepfake["timestamp"]
      latest_byte_string = face_swapper.current_deepfake["byte_string"]
      if latest_byte_string is not None:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_byte_string + b'\r\n')
    else:
      if latest_byte_string:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_byte_string + b'\r\n')
      else:
        time.sleep(0.001)


def run_flask(face_swapper, opts):
  """Define the app, and wrap it in CORS handler and in socketio."""
  log(f"Running Flask app with {opts}", "flask")

  # Start a Flask app.
  app = flask.Flask(__name__, template_folder="templates")
  app.config["APPLICATION_ROOT"] = "/"
  app.config["TEMPLATES_AUTO_RELOAD"] = True
  app.config["PREFERRED_URL_SCHEME"] = 'http'
  CORS(app, support_credentials=True)

  # Wrap the app in a socketIO.
  socketio = SocketIO(app, cors_allowed_origins="*", aync_mode="eventlet")
  logging.getLogger('werkzeug').disabled = True


  @socketio.on('connect')
  def connect():
    logging.info('Client connected from %s', flask.request.remote_addr)


  @socketio.on('disconnect')
  def disconnect():
    logging.info('Client %s disconnected', flask.request.remote_addr)


  @socketio.on('status')
  def status(data):
    """Callback for the socketIO returning the current state of narration.""" 
    nonlocal face_swapper
    try:
      socketio.emit('status-update', face_swapper.status())
      pass
    except Exception as e:
      logging.error(f"Error emitting stream status: {e}")
      raise e


  @app.route("/")
  @cross_origin(supports_credentials=True)
  def index():
    with open("templates/index.html", "r") as f:
      html_ui = f.read()
    return html_ui


  @app.route("/stage")
  @cross_origin(supports_credentials=True)
  def stage():
    with open("templates/index_stage.html", "r") as f:
      html_ui = f.read()
    return html_ui


  @app.route("/ui")
  @cross_origin(supports_credentials=True)
  def ui():
    with open("templates/ui.html", "r") as f:
      html_ui = f.read()
    return html_ui


  @app.route('/<filename>.js')
  @cross_origin(supports_credentials=True)
  def return_js(filename):
    """Function for returning Javascript."""
    filename = './templates/' + filename + '.js'
    return flask.send_file(filename, download_name=filename, mimetype='text/javascript')


  @app.route('/<filename>.css')
  @cross_origin(supports_credentials=True)
  def return_css(filename):
    """Function for returning CSS."""
    filename = './templates/' + filename + '.css'
    return flask.send_file(filename, download_name=filename, mimetype='text/css')


  @app.route('/images/<filename>')
  @cross_origin(supports_credentials=True)
  def return_image(filename):
    """Function for returning images."""
    import os
    filepath = os.path.abspath(os.path.join('images', filename))
    return flask.send_file(filepath, mimetype='image/jpeg')


  @app.route("/copy")
  @cross_origin(supports_credentials=True)
  def copy():
    nonlocal face_swapper
    face_swapper.read_source_image_from_file()
    return str(source_image["timestamp"])


  @app.route("/use_image/<index>")
  @cross_origin(supports_credentials=True)
  def use_image(index):
    nonlocal face_swapper
    import shutil
    try:
      if index == "camera":
        shutil.copyfile(_CAMERA_IMAGE_PATH, _TEMPORARY_IMAGE_PATH)
      else:
        shutil.copyfile(f"images/search_{index}.jpg", _TEMPORARY_IMAGE_PATH)
      face_swapper.read_source_image_from_file()
      return flask.jsonify({"status": "success"})
    except Exception as e:
      return flask.jsonify({"status": "error", "message": str(e)})


  @app.route("/click")
  @cross_origin(supports_credentials=True)
  def click():
    nonlocal face_swapper
    face_swapper.capture_source_image_from_camera()
    return str(face_swapper.source_image["timestamp"])


  @app.route("/active")
  @cross_origin(supports_credentials=True)
  def active():
    nonlocal face_swapper
    face_swapper.current_deepfake["active"] = True
    return str("active")


  @app.route("/inactive")
  @cross_origin(supports_credentials=True)
  def inactive():
    nonlocal face_swapper
    face_swapper.current_deepfake["active"] = False
    return str("inactive")


  @app.route("/many_faces")
  @cross_origin(supports_credentials=True)
  def many_faces():
    nonlocal face_swapper
    face_swapper.many_faces(True)
    return str("many_faces")


  @app.route("/single_face")
  @cross_origin(supports_credentials=True)
  def single_face():
    nonlocal face_swapper
    face_swapper.many_faces(False)
    face_swapper.reset_target_embedding()
    return str("single_face")


  @app.route("/reset_target")
  @cross_origin(supports_credentials=True)
  def reset_target():
    nonlocal face_swapper
    face_swapper.reset_target_embedding()
    return str("reset_target")


  @app.route("/background_removal_on")
  @cross_origin(supports_credentials=True)
  def background_removal_on():
    nonlocal face_swapper
    face_swapper.background_removal = True
    return str("background_removal_on")


  @app.route("/background_removal_off")
  @cross_origin(supports_credentials=True)
  def background_removal_off():
    nonlocal face_swapper
    face_swapper.background_removal = False
    return str("background_removal_off")


  @app.route("/search/<query>", methods=['GET'])
  @cross_origin(supports_credentials=True)
  def search(query):
    try:
      with ddgs.DDGS() as ddgs_search:
        results = ddgs_search.images(
            query=query,
            region="wt-wt",
            safesearch="moderate",
            max_results=opts.num_search_images
        )

      downloaded = []
      for index, result in enumerate(results):
        image_url = result.get('image')
        try:
          response = requests.get(image_url, timeout=10)
          response.raise_for_status()
          img = utils.get_image_from_bytes(response.content)
          
          if face_swapper.current_camera.get("image") is not None:
            cam_shape = face_swapper.current_camera["image"].shape
            cam_width, cam_height = cam_shape[1], cam_shape[0]
            img = utils.resize_image(img, cam_width, cam_height)

          filename = f"images/search_{index}.jpg"
          img.save(filename, "JPEG")
          downloaded.append(filename)
          if len(downloaded) == opts.num_search_images:
            break
        except Exception as e:
          log(f"Could not download image {image_url}: {e}", "error")
      return flask.jsonify({"status": "success", "images": downloaded})
    except Exception as e:
      return flask.jsonify({"status": "error", "message": str(e)}), 500


  @app.route("/source")
  @cross_origin(supports_credentials=True)
  def source():
    nonlocal face_swapper
    return flask.Response(source_stream(face_swapper),
                          mimetype='multipart/x-mixed-replace; boundary=frame')


  @app.route("/stream")
  @cross_origin(supports_credentials=True)
  def stream():
    nonlocal face_swapper
    return flask.Response(deepfake_stream(face_swapper),
                          mimetype='multipart/x-mixed-replace; boundary=frame')


  # Start the Flask app with GET, POST and sockets, in client-agnostic mode.
  socketio.run(app, host='0.0.0.0', port=opts.port, debug=False, use_reloader=False)


if __name__ == '__main__':
  face_swapper = FaceSwapper(opts)
  try:
    run_flask(face_swapper, opts)
  except KeyboardInterrupt:
    log("Ctrl+C pressed. Exiting...", "info")
    sys.exit(0)
