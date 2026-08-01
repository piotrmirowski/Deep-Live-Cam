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

from deep_fake import FaceSwapper
import utils
from generator import Generator, GENERATOR_STEPS
import threading
import mimetypes


_TEMPORARY_IMAGE_PATH = "images/temp.jpg"
_CAMERA_IMAGE_PATH = "images/camera.jpg"
_KEYS = ['bbox', 'kps', 'gender', 'age']
_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
_MESSAGE_STAGE_PATH = "templates/message_stage.txt"


parser = argparse.ArgumentParser(description='Deep Fake server')
parser.add_argument('-s', '--source', help='select an source image', dest='source_path',
                    default="templates/einstein.jpg")
parser.add_argument('--port', help='Port', dest='port', type=int, default=8001)
parser.add_argument('--alex_server', help='A.L.Ex server', type=str, default="")
parser.add_argument('--device', help='webcam device', dest='device',
                    type=str, default="Integrated Webcam")
parser.add_argument('--camera_index', help='webcam device index', dest='camera_index',
                    type=int, default=None)
parser.add_argument('--width', help='width in pixels', dest='width',
                    type=int, default=960)
parser.add_argument('--height', help='height in pixels', dest='height',
                    type=int, default=540)
parser.add_argument('--num-search-images', help='number of images to search by ddgs', dest='num_search_images',
                    type=int, default=3)
parser.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory',
                    type=int, default=16)
parser.add_argument('--execution-provider', help='execution provider', dest='execution_provider',
                    default=['coreml'], choices=['coreml', 'cuda', 'cpu'], nargs='+')
opts, _ = parser.parse_known_args()


def source_stream(face_swapper: FaceSwapper):
  """Loop that streams the most recent image source."""

  latest_byte_string = None
  latest_timestamp = 0
  while True:
    if latest_timestamp < face_swapper.source_image["timestamp"]:
      face_swapper.ping_stream()
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
      latest_timestamp = face_swapper.current_deepfake["timestamp"]
      latest_byte_string = face_swapper.current_deepfake["byte_string"]
      face_swapper.ping_stream()
      if latest_byte_string is not None:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_byte_string + b'\r\n')
    else:
      if latest_byte_string:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_byte_string + b'\r\n')
        time.sleep(0.04)
      else:
        time.sleep(0.001)


def run_flask(face_swapper, opts):
  """Define the app, and wrap it in CORS handler and in socketio."""
  utils.log(f"Running Flask app with {opts}", "flask")
  
  generator_instance = None
  alex_server = getattr(opts, 'alex_server', '')
  face_swapper.alex_server = alex_server
  provider = opts.execution_provider[0] if opts.execution_provider else 'cpu'
  generator_instance = Generator(folder="images", execution_provider=provider)

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
    """Callback for the socketIO returning the combined deep-fake + generator state."""
    try:
      socketio.emit('status-update', _get_combined_status())
    except Exception as e:
      logging.error(f"Error emitting stream status: {e}")
      raise e


  @app.route("/")
  @cross_origin(supports_credentials=True)
  def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
      html_ui = f.read()
    return html_ui


  @app.route("/stage")
  @cross_origin(supports_credentials=True)
  def stage():
    with open("templates/index_stage.html", "r", encoding="utf-8") as f:
      html_ui = f.read()
    # Load the configurable stage message and inject it as an HTML placeholder.
    stage_message = ""
    if os.path.exists(_MESSAGE_STAGE_PATH):
      with open(_MESSAGE_STAGE_PATH, "r", encoding="utf-8") as mf:
        stage_message = mf.read().strip()
    html_ui = html_ui.replace("{{STAGE_MESSAGE}}", stage_message)
    return html_ui


  @app.route("/ui")
  @cross_origin(supports_credentials=True)
  def ui():
    with open("templates/ui.html", "r", encoding="utf-8") as f:
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
    """Function for returning images or other generated assets."""
    import os
    import mimetypes
    filepath = os.path.abspath(os.path.join('images', filename))
    mime_type, _ = mimetypes.guess_type(filepath)
    if not mime_type:
      mime_type = 'application/octet-stream'
    return flask.send_file(filepath, mimetype=mime_type)


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


  @app.route("/start_camera")
  @cross_origin(supports_credentials=True)
  def start_camera():
    nonlocal face_swapper
    face_swapper.start_camera()
    return str("camera_started")


  @app.route("/stop_camera")
  @cross_origin(supports_credentials=True)
  def stop_camera():
    nonlocal face_swapper
    face_swapper.stop_camera()
    return str("camera_stopped")


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


  def _get_generator_status():
    """Return a dict describing the current generator state (no Flask response)."""
    if generator_instance is None:
      return {"status": "idle"}

    gen_status = "idle"
    current_step = None

    if generator_instance._start_time is not None:
      gen_status = "processing"

      for step in GENERATOR_STEPS:
        if generator_instance._timestamps.get(step) == "failed":
          gen_status = "failed"
          current_step = step
          break

      if gen_status != "failed":
        for step in GENERATOR_STEPS:
          if not generator_instance._source_face_path and step == "swap":
            continue
          if generator_instance._timestamps.get(step) == 0:
            current_step = step
            break
        if current_step is None:
          gen_status = "completed"

    steps_progress = {step: generator_instance._timestamps.get(step, 0) for step in GENERATOR_STEPS}

    return {
      "status": gen_status,
      "current_step": current_step,
      "steps": steps_progress,
      "celebrity": generator_instance._celebrity,
      "show_title": generator_instance._show_title,
      "image_description": generator_instance._image_description or "",
      "voice_description": generator_instance._voice_description or "",
      "image": f"/images/{os.path.basename(generator_instance._image_filename)}" if generator_instance._image_filename else None,
      "videos": [f"/images/{os.path.basename(v)}" for v in generator_instance._videos_completed if v],
      "swapped_videos": [f"/images/{os.path.basename(s)}" for s in generator_instance._swaps_completed if s]
    }


  def _get_combined_status():
    """Return face_swapper status merged with generator status under 'generator' key."""
    combined = face_swapper.status()
    combined["generator"] = _get_generator_status()
    return combined


  def _perform_search(query):
    nonlocal face_swapper
    nonlocal generator_instance
    try:
      print(f"Searching for {query}")
      with ddgs.DDGS() as ddgs_search:
        results = ddgs_search.images(
            query="face portrait of " + query,
            region="wt-wt",
            safesearch="moderate",
            max_results=opts.num_search_images
        )

      downloaded = []
      for index, result in enumerate(results):
        image_url = result.get('image')
        try:
          headers = {"User-Agent": _USER_AGENT}
          response = requests.get(image_url, timeout=10, headers=headers)
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
          utils.log(f"Could not download image {image_url}: {e}", "error")
      if downloaded:
        face_swapper.search_image = f"{query}_{time.time()}"
        generator_instance.set_celebrity(query)
      return flask.jsonify({"status": "success", "images": downloaded})
    except Exception as e:
      return flask.jsonify({"status": "error", "message": str(e)}), 500


  @app.route("/search/<query>", methods=['GET'])
  @cross_origin(supports_credentials=True)
  def search(query):
    return _perform_search(query)


  @app.route("/search_external/<query>", methods=['GET'])
  @cross_origin(supports_credentials=True)
  def search_external(query):
    nonlocal alex_server
    origin = flask.request.headers.get('Origin') or flask.request.referrer or flask.request.remote_addr
    if origin:
      alex_server = origin
      face_swapper.alex_server = origin
      utils.log(f"Updated alex_server to: {alex_server}", "flask")
    return _perform_search(query)


  @app.route("/ping", methods=['GET'])
  @cross_origin(supports_credentials=True)
  def ping():
    return flask.jsonify(_get_combined_status())


  @app.route("/source")
  @cross_origin(supports_credentials=True)
  def source():
    nonlocal face_swapper
    return flask.Response(source_stream(face_swapper),
                          mimetype='multipart/x-mixed-replace; boundary=frame')


  @app.route("/generate_celebrity", methods=['GET'])
  @cross_origin(supports_credentials=True)
  def generate_celebrity():
    nonlocal generator_instance
    celebrity = flask.request.args.get('celebrity', '').strip()
    title = flask.request.args.get('title', '').strip()
    try:
      sentence = int(flask.request.args.get('sentence', '1').strip())
    except ValueError:
      sentence = 1
    
    if not celebrity or not title:
      return flask.jsonify({"status": "error", "message": "Missing celebrity or title parameters."}), 400
      
    if generator_instance is not None and not generator_instance.is_done():
      return flask.jsonify({"status": "error", "message": "A generation is already in progress."}), 400
      
    # Determine the source face path
    source_face_path = None
    if os.path.exists("images/temp.jpg"):
      source_face_path = "images/temp.jpg"
    elif opts.source_path and os.path.exists(opts.source_path):
      source_face_path = opts.source_path
      
    try:
      # Run generation in a background thread to prevent blocking Flask thread/loop
      thread = threading.Thread(
          target=generator_instance.generate,
          args=(celebrity, title, source_face_path, sentence),
          daemon=True
      )
      thread.start()
      return flask.jsonify({"status": "success", "message": "Generation started successfully."})
    except Exception as e:
      utils.log(f"Error starting generation: {e}", "error")
      return flask.jsonify({"status": "error", "message": str(e)}), 500


  @app.route("/generation_status", methods=['GET'])
  @cross_origin(supports_credentials=True)
  def generation_status():
    """Return the combined deep-fake + generator status."""
    return flask.jsonify(_get_combined_status())


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
    utils.log("Ctrl+C pressed. Exiting...", "info")
    sys.exit(0)
