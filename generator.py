import os
import sys
import time
import argparse
import logging
import threading
import shutil
import unicodedata
import re
from generation.llm import LLM
from generation.veo import Veo
import deep_fake

GENERATOR_STEPS = ["description", "voice_description", "image", "video", "swap"]
PROMPT_DESCRIPTION = (
    "You are writing a prompt for Gemini Nano Banana Pro.\n" +
    "Describe physically the following person in a very detailed manner, " +
    "mentioning their appearance and clothes (down to the color of the clothes, " +
    "and the shape of their glasses, if they are wearing glasses). " +
    "The person is [CELEBRITY], but do not mention their name. Write that description in a single paragraph. " +
    "Focus your description on the appearance from the waist up, as the person " +
    "is taking a selfie. The camera is not visible on the image, " +
    "and we cannot see the arm that is holding the phone either.\n" +
    "The image is a selfie-style portrait from a phone's front camera (f/4.6, 24mm). " +
    "The person is looking directly into the lens and smiling, recording a video message. " +
    "The person is sitting at the back of a luxury limousine with black leather seats. " +
    "The outside view is Edinburgh Castle in the distance and The Royal Mile, " +
    "with performers, street artists and passers-by walking along busy streets." +
    "The image has cinematic natural lighting.\n" +
    "Combine all these elements to write a prompt that contains: " +
    "subject, composition, action, lighting and background description.")
PROMPT_VOICE = (
    "You are writing a prompt for Veo 3.1 Flash, as a voice coach.\n" +
    "The person is [CELEBRITY]: describe the voice of [CELEBRITY], " +
    "with all its characteristics, mentioning their regional accent, specific dialect, texture, pitch, pace, energy, emotional tone, " +
    "and any other relevant features useful for a voice actor imitating their voice. " +
    "(Remember: do not mention the name of the celebrity in the voice description!)\n" +
    "Write that description in a single sentence of 20 words.")
SENTENCES = [
    ("Hello everyone! " +
     "I am on my way. In the meantime, you have two minutes to deliberate. " +
     "One of you has to go home."),
    ("Hey! Time's up. Cast your votes now.")
]
PROMPT_VIDEO = (
    "A high-quality selfie portrait video showing a person smiling, " +
    "looking directly at the camera, and speak naturally. Soft cinematic lighting, " +
    "subtle lifelike movements, matching composition from the image. " +
    "Selfie video portrait shot, from phone frontal camera (f/4.6, 24mm). " +
    "The person speaks the message: \"[MESSAGE]\"\n" +
    "The voice of the actor is described in these terms: [VOICE_DESCRIPTION]\n" +
    "The person is sitting at the back of a luxury limousine. " +
    "The outside view is Edinburgh Castle in the distance and The Royal Mile, " +
    "with performers, street artists and passers-by walking along busy streets."
)
VIDEO_ASPECT_RATIO = "9:16"
RESIZE_TO_1080 = True
DRIVE_FOLDER = r"c:\Users\pimir\My Drive\HumanMachine\Improbotics\Tech\Tech_Reality_2026\Impro_Show_Ctl\assets\vid"
DRIVE_FILENAMES = ["05 Celebrity 1.mp4", "06 Celebrity 2.mp4"]

class Generator:

  def __init__(self, folder: str = "images", execution_provider: str = "cuda"):
    self._base_folder = folder
    self._folder = folder
    self._execution_provider = execution_provider
    self._llm = LLM("gemini")
    self._veo = Veo()
    self._timestamps = {step: 0 for step in GENERATOR_STEPS}
    self.reset()


  def reset(self):
    self._folder = self._base_folder
    self._celebrity = ""
    self._show_title = ""
    self._source_face_path = None
    self._image_description = None
    self._voice_description = None
    self._image = None
    self._image_filename = None
    self._start_time = None
    self._sentence = 1
    self._timestamps = {step: 0 for step in GENERATOR_STEPS}
    self._videos_completed = [None] * len(SENTENCES)
    self._swaps_completed = [None] * len(SENTENCES)


  def set_celebrity(self, celebrity: str) -> None:
    self._celebrity = celebrity


  def _prompt_description(self, celebrity: str) -> str:
    return PROMPT_DESCRIPTION.replace("[CELEBRITY]", celebrity)


  def _prompt_voice_description(self, celebrity: str) -> str:
    return PROMPT_VOICE.replace("[CELEBRITY]", celebrity)


  def _log_to_file(self, step: str, prompt: str, response: str, identifier: str):
    os.makedirs(self._folder, exist_ok=True)
    log_file_path = os.path.join(self._folder, f"{step}_{identifier}.log")
    with open(log_file_path, "w", encoding="utf-8") as f:
      f.write(f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response}\n")


  def _save_image(self, step: str, PIL_image, identifier: str) -> str:
    os.makedirs(self._folder, exist_ok=True)
    image_file_path = os.path.join(self._folder, f"{step}_{identifier}.png")
    PIL_image.save(image_file_path)
    return image_file_path


  def _generate_description(self, timestamp: float):
    step = f"description"
    identifier = f"{timestamp:.1f}"

    def callback_description(prompt, response):
      if isinstance(response, dict) and "error" in response:
        logging.error(f"Error generating description: {response['error']}")
        self._timestamps[step] = "failed"
        self._timestamps["voice_description"] = "failed"
        self._timestamps["image"] = "failed"
        self._timestamps["video"] = "failed"
        if self._source_face_path:
          self._timestamps["swap"] = "failed"
        return

      self._log_to_file(step, prompt, str(response), identifier)
      self._timestamps[step] = identifier
      if isinstance(response, dict):
        self._image_description = response.get("text", "")
      else:
        self._image_description = response

      logging.info(f"Description generated successfully! Saved log to {self._folder}.")
      print(f"Description:\n{self._image_description}\n")
      logging.info("Starting voice description generation...")
      self._generate_voice_description(timestamp)

    prompt = self._prompt_description(self._celebrity)
    self._llm.schedule_generate(prompt,
                                callback_description,
                                response_format=None,
                                max_tokens=1024,
                                thinking=True)


  def _generate_voice_description(self, timestamp: float):
    step = f"voice_description"
    identifier = f"{timestamp:.1f}"

    def callback_voice_description(prompt, response):
      if isinstance(response, dict) and "error" in response:
        logging.error(f"Error generating voice description: {response['error']}")
        self._timestamps[step] = "failed"
        self._timestamps["image"] = "failed"
        self._timestamps["video"] = "failed"
        if self._source_face_path:
          self._timestamps["swap"] = "failed"
        return

      self._log_to_file(step, prompt, str(response), identifier)
      self._timestamps[step] = identifier
      if isinstance(response, dict):
        self._voice_description = response.get("text", "")
      else:
        self._voice_description = response

      logging.info(f"Voice description generated successfully! Saved log to {self._folder}.")
      print(f"Voice Description:\n{self._voice_description}\n")
      logging.info("Starting image generation...")
      self._generate_image(timestamp)

    prompt = self._prompt_voice_description(self._celebrity)
    self._llm.schedule_generate(prompt,
                                callback_voice_description,
                                response_format=None,
                                max_tokens=1024,
                                thinking=True)


  def _generate_image(self, timestamp: float):
    step = f"image"
    identifier = f"{timestamp:.1f}"

    def callback_image(prompt, response):
      if isinstance(response, dict) and "error" in response:
        logging.error(f"Error generating image: {response['error']}")
        self._timestamps[step] = "failed"
        self._timestamps["video"] = "failed"
        if self._source_face_path:
          self._timestamps["swap"] = "failed"
        return

      text = response["text"] if "text" in response else ""
      self._log_to_file(step, prompt, text, identifier)
      self._timestamps[step] = identifier
      if "image" in response:
        filename = self._save_image(step, response["image"], identifier)
        self._image = response["image"]
        self._image_filename = filename
        print(f"Image generated successfully! Saved to {filename}")
        logging.info("Starting pipelined video generation with Veo...")
        for i in range(len(SENTENCES)):
          self._generate_video_for_sentence(i, timestamp)
      else:
        logging.warning("No image data found in model response.")
        self._timestamps["video"] = "failed"
        if self._source_face_path:
          self._timestamps["swap"] = "failed"

    prompt = self._image_description
    self._llm.schedule_generate(prompt,
                                callback_image,
                                response_format={"image_size": "1K", "aspect_ratio": VIDEO_ASPECT_RATIO},
                                is_image=True)


  def _generate_video_for_sentence(self, index: int, timestamp: float):
    step = f"video"
    identifier = f"{timestamp:.1f}"

    def callback_video(prompt, response):
      if isinstance(response, dict) and "error" in response:
        logging.error(f"Error generating video {index+1}: {response['error']}")
        self._timestamps[step] = "failed"
        if self._source_face_path:
          self._timestamps["swap"] = "failed"
        return

      try:
        generated_video = response["video"]
        client = response["client"]
        video_file_path = os.path.join(self._folder, f"{step}_{index+1}_{identifier}.mp4")

        logging.info(f"Downloading and saving the generated video {index+1}...")
        client.files.download(file=generated_video.video)
        generated_video.video.save(video_file_path)

        self._videos_completed[index] = video_file_path
        print(f"Video {index+1} generated successfully! Saved to {video_file_path}")

        # If we are not doing face swap:
        if not self._source_face_path:

          # Check if all videos are completed
          if all(v is not None for v in self._videos_completed):
            self._timestamps["video"] = identifier
            logging.info("All video generations completed successfully!")
        else:
          # Trigger face swapping for this video
          self._generate_swap_for_sentence(index, video_file_path, timestamp)

        # # Start generating the next video while processing the swap / continuation
        # next_index = index + 1
        # if next_index < len(SENTENCES):
        #   logging.info(f"Pipelining: Starting generation for video {next_index+1}...")
        #   self._generate_video_for_sentence(next_index, timestamp)

      except Exception as e:
        logging.error(f"Error saving generated video {index+1}: {e}")
        self._timestamps[step] = "failed"
        if self._source_face_path:
          self._timestamps["swap"] = "failed"

    # Format the prompt with the specific sentence text
    sentence_text = SENTENCES[index]
    video_prompt = PROMPT_VIDEO.replace("[MESSAGE]", sentence_text).replace("[VOICE_DESCRIPTION]", self._voice_description)
    print(f"Prompt for video {index+1}: {video_prompt}")

    def run_veo():
      self._veo.generate_video(
          prompt=video_prompt,
          image_path=self._image_filename,
          callback=callback_video
      )

    # Execute Veo operation in a background thread to prevent blocking client loop
    thread = threading.Thread(target=run_veo, daemon=True)
    thread.start()


  def _generate_swap_for_sentence(self, index: int, video_path: str, timestamp: float):
    step = f"swap"
    identifier = f"{timestamp:.1f}"

    def run_swap():
      try:
        swapped_video_path = os.path.join(self._folder, f"{step}_{index+1}_{identifier}.mp4")
        logging.info(f"Swapping face from '{self._source_face_path}' into video {index+1} '{video_path}'...")
        
        deep_fake.swap_video(
            source_path=self._source_face_path,
            target_path=video_path,
            output_path=swapped_video_path,
            execution_provider=self._execution_provider,
            verbose=True,
            resize_to_1080=RESIZE_TO_1080
        )

        self._swaps_completed[index] = swapped_video_path
        logging.info(f"Face swapping for video {index+1} completed successfully! Saved to {swapped_video_path}")

        # Copy face-swapped video to Google Drive
        self._copy_to_drive(swapped_video_path, index)

        # Check if all swaps are completed
        if all(s is not None for s in self._swaps_completed):
          self._timestamps["video"] = identifier
          self._timestamps["swap"] = identifier
          logging.info("All face swaps and generations completed successfully!")
      except Exception as e:
        logging.error(f"Error during face swapping for video {index+1}: {e}")
        self._timestamps[step] = "failed"

    # Execute face-swapping in a background thread to keep execution non-blocking
    thread = threading.Thread(target=run_swap, daemon=True)
    thread.start()


  def _copy_to_drive(self, result_path: str, index: int):
    import shutil
    try:
      dest_filename = DRIVE_FILENAMES[index]
      dest_path = os.path.join(DRIVE_FOLDER, dest_filename)
      os.makedirs(DRIVE_FOLDER, exist_ok=True)
      logging.info(f"Copying final video {index+1} from '{result_path}' to '{dest_path}'...")
      shutil.copy(result_path, dest_path)
      logging.info(f"Copying video {index+1} to Drive succeeded.")
    except Exception as e:
      logging.error(f"Failed to copy video {index+1} to Drive: {e}")


  def _load_from_cache(self):
    """Loads results from existing celebrity folder cache, copies temp.jpg, and copies videos to Drive."""
    # 1. Copy CELEBRITY_CLEAN/temp.jpg onto temp.jpg (e.g. images/temp.jpg)
    cached_temp = os.path.join(self._folder, "temp.jpg")
    if os.path.exists(cached_temp):
      dst_temp = os.path.join(self._base_folder, "temp.jpg")
      try:
        shutil.copy(cached_temp, dst_temp)
        logging.info(f"Copied '{cached_temp}' to '{dst_temp}'.")
      except Exception as e:
        logging.error(f"Failed to copy '{cached_temp}' to '{dst_temp}': {e}")
      self._source_face_path = cached_temp

    # 2. Extract timestamp and find generated files in folder
    identifier = None
    files = os.listdir(self._folder) if os.path.exists(self._folder) else []

    # First look for timestamp in image or log or video filenames
    for filename in files:
      m = re.search(r"(?:image|description|voice_description|video_\d+|swap_\d+)_([0-9.]+)\.(?:png|log|mp4)", filename)
      if m:
        identifier = m.group(1)
        break

    if identifier:
      try:
        self._start_time = float(identifier)
      except ValueError:
        self._start_time = time.time()
    else:
      self._start_time = time.time()
      identifier = f"{self._start_time:.1f}"

    # Load image
    for filename in files:
      if filename.startswith("image_") and filename.endswith(".png"):
        self._image_filename = os.path.join(self._folder, filename)
        self._timestamps["image"] = identifier
        try:
          from PIL import Image
          self._image = Image.open(self._image_filename)
        except Exception:
          pass
        break

    # Load description
    for filename in files:
      if filename.startswith("description_") and filename.endswith(".log"):
        desc_path = os.path.join(self._folder, filename)
        self._timestamps["description"] = identifier
        try:
          with open(desc_path, "r", encoding="utf-8") as f:
            content = f.read()
            if "RESPONSE:\n" in content:
              self._image_description = content.split("RESPONSE:\n", 1)[1].strip()
            else:
              self._image_description = content.strip()
        except Exception as e:
          logging.error(f"Failed to read description log: {e}")
        break

    # Load voice description
    for filename in files:
      if filename.startswith("voice_description_") and filename.endswith(".log"):
        voice_path = os.path.join(self._folder, filename)
        self._timestamps["voice_description"] = identifier
        try:
          with open(voice_path, "r", encoding="utf-8") as f:
            content = f.read()
            if "RESPONSE:\n" in content:
              self._voice_description = content.split("RESPONSE:\n", 1)[1].strip()
            else:
              self._voice_description = content.strip()
        except Exception as e:
          logging.error(f"Failed to read voice description log: {e}")
        break

    # Load videos
    for filename in files:
      m = re.match(r"^video_(\d+)_.*\.mp4$", filename)
      if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(SENTENCES):
          self._videos_completed[idx] = os.path.join(self._folder, filename)

    if any(v is not None for v in self._videos_completed):
      self._timestamps["video"] = identifier

    # Load swapped videos
    for filename in files:
      m = re.match(r"^swap_(\d+)_.*\.mp4$", filename)
      if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(SENTENCES):
          self._swaps_completed[idx] = os.path.join(self._folder, filename)

    if any(s is not None for s in self._swaps_completed):
      self._timestamps["swap"] = identifier

    logging.info(f"Loaded cached state for {self._celebrity}: image={self._image_filename}, videos={self._videos_completed}, swaps={self._swaps_completed}")

    # 3. Copy to drive
    if any(s is not None for s in self._swaps_completed):
      for idx, s_path in enumerate(self._swaps_completed):
        if s_path and os.path.exists(s_path):
          self._copy_to_drive(s_path, idx)
    elif any(v is not None for v in self._videos_completed):
      for idx, v_path in enumerate(self._videos_completed):
        if v_path and os.path.exists(v_path):
          self._copy_to_drive(v_path, idx)


  def generate(self, celebrity: str, show_title: str, source_face_path: str = None, sentence: int = 1):
    self.reset()
    self._celebrity = celebrity
    self._show_title = show_title
    self._sentence = sentence

    nfkd = unicodedata.normalize('NFKD', celebrity)
    celebrity_clean = "".join(c for c in nfkd if ("a" <= c <= "z" or "A" <= c <= "Z"))
    if celebrity_clean:
      target_folder = os.path.join(self._base_folder, celebrity_clean)
    else:
      target_folder = self._base_folder

    if celebrity_clean and os.path.isdir(target_folder):
      self._folder = target_folder
      logging.info(f"Normalized celebrity folder '{self._folder}' exists. Loading from cache...")
      self._load_from_cache()
      return

    self._folder = target_folder
    os.makedirs(self._folder, exist_ok=True)

    if source_face_path and os.path.exists(source_face_path):
      copied_source_path = os.path.join(self._folder, os.path.basename(source_face_path))
      if os.path.abspath(source_face_path) != os.path.abspath(copied_source_path):
        try:
          shutil.copy(source_face_path, copied_source_path)
          logging.info(f"Copied source face image from '{source_face_path}' to '{copied_source_path}'.")
        except Exception as e:
          logging.error(f"Failed to copy source face image: {e}")
      self._source_face_path = copied_source_path
    else:
      self._source_face_path = source_face_path

    self._start_time = time.time()
    logging.info(f"Starting generation process for celebrity: {celebrity} in folder: {self._folder}")
    self._generate_description(self._start_time)


  def is_done(self) -> bool:
    if self._start_time is None:
      return True
    if self._source_face_path:
      return self._timestamps["swap"] != 0 or self._timestamps["swap"] == "failed"
    return self._timestamps["video"] != 0 or self._timestamps["video"] == "failed"


def main():
  parser = argparse.ArgumentParser(description="CLI to generate physical description, image, and video of a celebrity using LLM.")
  parser.add_argument('-c', '--celebrity', help='Name of the celebrity (e.g. "Tom Cruise")', required=True)
  parser.add_argument('-t', '--title', help='Show title', required=True)
  parser.add_argument('-s', '--source', help='Path to source face image to swap into the generated video (optional)', default=None)
  parser.add_argument('-f', '--folder', default='images', help='Folder where to store logs, image, and video outputs')
  parser.add_argument('--sentence', type=int, choices=[1, 2], default=1, help='Sentence template index (1 or 2)')
  parser.add_argument('--verbose', action='store_true', help='Enable verbose debug logging')

  args = parser.parse_args()

  log_level = logging.DEBUG if args.verbose else logging.INFO
  logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')

  generator = Generator(folder=args.folder)
  generator.generate(args.celebrity, args.title, args.source, sentence=args.sentence)

  # Since the generation runs in the background in scheduling threads, poll for completion
  try:
    while not generator.is_done():
      time.sleep(0.5)
  except KeyboardInterrupt:
    logging.info("\nGeneration aborted by user.")
    sys.exit(1)

  failed_step = None
  for step in GENERATOR_STEPS:
    if step == "swap" and not args.source:
      continue
    if generator._timestamps[step] == "failed":
      failed_step = step
      break

  if failed_step:
    logging.error(f"Generation failed during '{failed_step}' processing stage.")
    sys.exit(1)
  else:
    logging.info("Generation finished successfully.")
    if args.source:
      completed = [s for s in generator._swaps_completed if s is not None]
      logging.info(f"Final face-swapped videos: {completed}")
    else:
      completed = [v for v in generator._videos_completed if v is not None]
      logging.info(f"Generated videos: {completed}")


if __name__ == "__main__":
  main()
