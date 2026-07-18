import os
import time
import threading
import cv2
import numpy as np
import rembg
import pyvirtualcam

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


class FaceSwapper(object):

  def __init__(self, opts):
    self._last_timestamp_stream = 0

    # Make sure the images directory exists
    if not os.path.exists("images"):
      os.makedirs("images")

    # Initialise the parameters
    self._source_path = opts.source_path
    cli_mode = getattr(opts, 'cli_mode', False)
    if not cli_mode:
      if getattr(opts, 'camera_index', None) is not None:
        utils.list_webcams(opts.device)
        self._device = opts.camera_index
        utils.log(f"Using camera index {self._device} specified by --camera-index", "info")
      else:
        self._device = utils.list_webcams(opts.device)
    else:
      self._device = None
    self._width = opts.width
    self._height = opts.height
    self._init(opts)

    # Current image and deepfake storage
    self.source_image = {"image": None, "annotated_image": None, "timestamp": 0}
    self.current_camera = {"image": None, "byte_string": None, "timestamp": 0}
    self.current_deepfake = {"image": None, "byte_string":None, "timestamp": 0, "active": False}
    self.current_faces = []
    self.target_embedding = None
    self.camera_streaming = True
    self.search_image = None

    if not cli_mode:
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
    if not cli_mode:
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


  def ping_stream(self):
    self._last_timestamp_stream = time.time()


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
        utils.log("Could not pre-check frame_processor", "error")
        exit(1)
    core.limit_resources()


  def status(self):
    return {"many_faces": modules.globals.many_faces,
            "faces": self.current_faces,
            "active": self.current_deepfake["active"],
            "background_removal": self.background_removal,
            "camera_streaming": self.camera_streaming,
            "search_image": self.search_image}

  def start_camera(self):
    if not self.camera_streaming:
      utils.log("Starting camera...", "info")
      self._cap = cv2.VideoCapture(self._device)
      self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
      self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
      self._cap.set(cv2.CAP_PROP_FPS, 60)
      self.camera_streaming = True

  def stop_camera(self):
    if self.camera_streaming:
      utils.log("Stopping camera...", "info")
      self.camera_streaming = False
      if self._cap is not None:
        self._cap.release()
        self._cap = None
      # Generate placeholder frame when camera stops
      placeholder = np.zeros((self._height, self._width, 3), dtype=np.uint8)
      cv2.putText(placeholder, "Camera Stopped", (self._width // 2 - 150, self._height // 2),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
      self.current_camera["image"] = placeholder
      self.current_camera["timestamp"] = time.time()
      self.current_camera["byte_string"] = utils.write_numpy_to_byte_string(placeholder)

      self.current_deepfake["image"] = placeholder
      self.current_deepfake["byte_string"] = utils.write_numpy_to_byte_string(placeholder)
      self.current_deepfake["timestamp"] = time.time()


  def _store_source_image(self, cv2_image):
    if cv2_image is not None:
      utils.log(f"Image of type {type(cv2_image)}, shape {cv2_image.shape}, max {cv2_image.max()}", "source")
      self.source_image["image"] = cv2_image
      self.source_image["annotated_image"] = get_one_face(cv2_image)
      self.source_image["byte_string"] = utils.write_numpy_to_byte_string(self.source_image["image"])
      self.source_image["timestamp"] = time.time()


  def read_source_image_from_file(self) -> None:
    """Read the source image from a file."""
    utils.log(f"Reading image from {_TEMPORARY_IMAGE_PATH}...", "source")
    cv2_image = cv2.imread(_TEMPORARY_IMAGE_PATH)
    self._store_source_image(cv2_image)


  def capture_source_image_from_camera(self) -> None:
    """Capture the source image from the camera."""
    if self.current_camera["image"] is not None:
      utils.log(f"Capturing camera image, storing in {_CAMERA_IMAGE_PATH}...", "source")
      cv2_image = self.current_camera["image"].copy()
      cv2.imwrite(_CAMERA_IMAGE_PATH, cv2_image)
      cv2.imwrite(_TEMPORARY_IMAGE_PATH, cv2_image)
      self._store_source_image(cv2_image)


  def _load_source_image_from_file(self) -> None:
    """Load the source image from a file."""
    if self._source_path:
      utils.log(f"Loading image {self._source_path}...", "source")
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
      utils.log("No source face found.", "error")
      return temp_frame

    # Get all the faces in the temp frame.
    many_faces = get_many_faces(temp_frame)
    self._store_face_stats(many_faces)


    # Early exit if no target face is found.
    if not many_faces:
      utils.log("No target faces found.", "error")
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
    last_stream_time = self._last_timestamp_stream
    delta_t_stream = 0
    stream_fps = 0

    while True:
      if not self.camera_streaming or self._cap is None:
        time.sleep(0.1)
        continue

      # Read the camera and crash if no image.
      camera_return, camera_buffer = self._cap.read()
      if not camera_return:
        utils.log("Cannot get camera input.", "error")
        if not self.camera_streaming:
          continue
        time.sleep(0.1)
        continue
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

      # Compute streaming FPS
      if self._last_timestamp_stream - time.time() < 5:
        if self._last_timestamp_stream != last_stream_time:
          dt = self._last_timestamp_stream - last_stream_time
          delta_t_stream = 0.5 * dt + delta_t_stream * 0.5
          last_stream_time = self._last_timestamp_stream
          stream_fps = 1.0 / (delta_t_stream + 1e-8)
      else:
        stream_fps = 0

      print(f"[info] \033[36mFPS: {1/delta_t_frame:.2f}\033[0m, \033[31mstream FPS: {stream_fps:.2f}\033[0m, \033[32msize: {delta_t_size:.2f}s\033[0m, \033[33mbg: {delta_t_bg:.2f}s\033[0m, \033[35mswap: {delta_t_swap:.2f}s\033[0m\033[K", end='\r', flush=True)


  def start(self):
    self._thread = threading.Thread(target=self._run_deep_fake_loop, args=(), daemon=True)
    self._thread.start()


def swap_video(source_path: str,
               target_path: str,
               output_path: str,
               execution_provider: str = 'cuda',
               max_memory: int = None,
               verbose: bool = True,
               resize_to_1080: bool = False) -> None:
  """
  Perform face swapping on all frames of the target video using the source face image.
  Saves the result to output_path and merges the original audio.

  Args:
    source_path (str): Path to the source face image (e.g. images/temp.jpg)
    target_path (str): Path to the target video (e.g. images/Man_speaks_to_camera_202605221602.mp4)
    output_path (str): Path to save the processed video (e.g. images/output.mp4)
    execution_provider (str): ONNX execution provider (e.g. cpu, cuda, directml)
    max_memory (int, optional): Maximum amount of RAM in GB. If None, suggests max memory automatically.
    verbose (bool): If True, prints progress details to stdout.
    resize_to_1080 (bool): If True, resizes and reshapes to 1920x1080 (adds black bars on the side if vertical).
  """

  # Check paths.
  if not os.path.exists(source_path):
    raise FileNotFoundError(f"Source image file '{source_path}' does not exist.")
  if not os.path.exists(target_path):
    raise FileNotFoundError(f"Target video file '{target_path}' does not exist.")
  output_dir = os.path.dirname(os.path.abspath(output_path))
  if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

  class FaceSwapperOpts:
    def __init__(self):
      self.source_path = source_path
      self.width = 960
      self.height = 540
      self.max_memory = max_memory if max_memory is not None else core.suggest_max_memory()
      self.execution_provider = [execution_provider]
      self.cli_mode = True
  opts = FaceSwapperOpts()

  if verbose:
    print("--------------------------------------------------")
    print("Initializing FaceSwapper...")
    print(f"Source Image: {source_path}")
    print(f"Target Video: {target_path}")
    print(f"Output Video: {output_path}")
    print(f"Execution Provider: {execution_provider}")
    print("--------------------------------------------------")
  swapper = FaceSwapper(opts)
  swapper.many_faces(True)

  # Get the annotated source face.
  source_face = swapper.source_image["annotated_image"]
  if not source_face:
    raise ValueError(f"No face detected in the source image '{source_path}'.")
  # Open target video.
  cap = cv2.VideoCapture(target_path)
  if not cap.isOpened():
    raise IOError(f"Could not open target video '{target_path}' using OpenCV.")

  # Get video metadata
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if total_frames <= 0:
    total_frames = 1
  if verbose:
    print(f"Target Video Resolution: {width}x{height}")
    print(f"Target Video FPS: {fps:.2f}")
    print(f"Total Frames to Process: {total_frames}")
    print("--------------------------------------------------")

  # Determine final output dimensions
  out_width = width
  out_height = height
  if resize_to_1080:
    out_width = 1920
    out_height = 1080

  # Initialize video writer.
  temp_output = output_path + ".temp.mp4"
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(temp_output, fourcc, fps, (out_width, out_height))
  if not out.isOpened():
    cap.release()
    raise IOError(f"Could not open output video writer for '{output_path}'.")

  start_time = time.time()
  processed_count = 0
  try:
    if verbose:
      utils.print_progress_bar(0, total_frames, prefix='Processing:', suffix='Complete', length=40)
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      # Swap faces in the frame
      processed_frame = swapper._process_frame(source_face, frame)

      # Optional resize and reshape to 1920x1080
      if resize_to_1080:
        h, w = processed_frame.shape[:2]
        if h > w:
          # Vertical mode: scale proportionally to height = 1080, pad sides with black bars
          scale = 1080.0 / h
          new_w = int(w * scale)
          new_w = max(1, new_w)
          resized_sub = cv2.resize(processed_frame, (new_w, 1080))
          
          # Create black background and center the frame
          black_bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
          x_offset = (1920 - new_w) // 2
          black_bg[:, x_offset:x_offset + new_w] = resized_sub
          processed_frame = black_bg
        else:
          # Landscape or square: resize directly to 1920x1080
          processed_frame = cv2.resize(processed_frame, (1920, 1080))

      # Write the processed frame to output video file
      out.write(processed_frame)

      processed_count += 1
      if verbose:
        elapsed = time.time() - start_time
        fps_speed = processed_count / elapsed if elapsed > 0 else 0
        suffix = f'{processed_count}/{total_frames} ({fps_speed:.1f} fps)'
        utils.print_progress_bar(processed_count, total_frames, prefix='Processing:', suffix=suffix, length=40)

  except KeyboardInterrupt as e:
    if verbose:
      print("\nProcess interrupted by user.")
    raise e
  except Exception as e:
    if verbose:
      print(f"\nAn error occurred during face swapping: {e}")
    raise e
  finally:
    cap.release()
    out.release()

  total_time = time.time() - start_time
  if verbose:
    print("--------------------------------------------------")
    print(f"Face swapping completed in {total_time:.2f} seconds.")
  
  # Merge original audio from the target video into the face-swapped video
  utils.merge_audio(temp_output, target_path, output_path)

  if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
    raise IOError("Output video file is empty or was not created.")

  if verbose:
    print(f"Successfully saved output video to: {output_path}")
    print(f"Output File Size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    print("--------------------------------------------------")


def swap_image(source_path: str,
               target_path: str,
               output_path: str,
               execution_provider: str = 'cuda',
               max_memory: int = None,
               verbose: bool = True) -> None:
  """
  Perform face swapping on a single target image using the source face image.
  Saves the result to output_path.

  Args:
    source_path (str): Path to the source face image (e.g. images/temp.jpg)
    target_path (str): Path to the target image (e.g. images/target.jpg)
    output_path (str): Path to save the processed image (e.g. images/output.jpg)
    execution_provider (str): ONNX execution provider (e.g. cpu, cuda, directml)
    max_memory (int, optional): Maximum amount of RAM in GB. If None, suggests max memory automatically.
    verbose (bool): If True, prints progress details to stdout.
  """
  # Check paths.
  if not os.path.exists(source_path):
    raise FileNotFoundError(f"Source image file '{source_path}' does not exist.")
  if not os.path.exists(target_path):
    raise FileNotFoundError(f"Target image file '{target_path}' does not exist.")
  output_dir = os.path.dirname(os.path.abspath(output_path))
  if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

  class FaceSwapperOpts:
    def __init__(self):
      self.source_path = source_path
      self.width = 960
      self.height = 540
      self.max_memory = max_memory if max_memory is not None else core.suggest_max_memory()
      self.execution_provider = [execution_provider]
      self.cli_mode = True
  opts = FaceSwapperOpts()

  if verbose:
    print("--------------------------------------------------")
    print("Initializing FaceSwapper for image...")
    print(f"Source Image: {source_path}")
    print(f"Target Image: {target_path}")
    print(f"Output Image: {output_path}")
    print(f"Execution Provider: {execution_provider}")
    print("--------------------------------------------------")
  swapper = FaceSwapper(opts)
  swapper.many_faces(True)

  # Get the annotated source face.
  source_face = swapper.source_image["annotated_image"]
  if not source_face:
    raise ValueError(f"No face detected in the source image '{source_path}'.")

  # Read target image.
  target_image = cv2.imread(target_path)
  if target_image is None:
    raise IOError(f"Could not open target image '{target_path}' using OpenCV.")

  start_time = time.time()
  try:
    if verbose:
      print("Processing image...")
    # Swap faces in the target image
    processed_image = swapper._process_frame(source_face, target_image)
    # Write the processed image to output
    cv2.imwrite(output_path, processed_image)
  except Exception as e:
    if verbose:
      print(f"\nAn error occurred during face swapping: {e}")
    raise e

  total_time = time.time() - start_time
  if verbose:
    print("--------------------------------------------------")
    print(f"Face swapping completed in {total_time:.2f} seconds.")

  if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
    raise IOError("Output image file is empty or was not created.")

  if verbose:
    print(f"Successfully saved output image to: {output_path}")
    print(f"Output File Size: {os.path.getsize(output_path) / 1024:.2f} KB")
    print("--------------------------------------------------")

