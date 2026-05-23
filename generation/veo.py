from collections.abc import Callable
import logging
import os
import time

from google import genai
from google.genai import types

from .generic_llm import GenericLLM

VEO_MODEL_NAME = "veo-3.1-generate-preview"


class Veo(GenericLLM):
  """Class interfacing with Google's Veo video generation models."""

  def __init__(self, model_name: str = VEO_MODEL_NAME):
    super().__init__()
    api_key = os.getenv("GEMINI_API_KEY")
    assert api_key, "Please set GEMINI_API_KEY environment variable"
    self._model_name = model_name
    self._client = genai.Client(api_key=api_key)
    logging.info("Configured Veo with GEMINI_API_KEY=%s", api_key)

  def generate_video(self,
                     prompt: str,
                     image_path: str,
                     callback: Callable = None):
    """
    Generate a video using the Veo model starting with a reference image and a prompt.
    This runs synchronously within this function, so it should be scheduled/run on a background thread.
    """
    try:
      self._time_scheduled = time.time()

      # 1. Initiate video generation by passing types.Image directly
      logging.info(f"Initiating video generation with Veo ({self._model_name})...")
      operation = self._client.models.generate_videos(
          model=self._model_name,
          prompt=prompt,
          config=types.GenerateVideosConfig(
              aspect_ratio="9:16",
              resolution="720p",
          ),
          image=types.Image.from_file(location=image_path)
      )

      # 3. Poll operation status until completion
      logging.info("Veo generation queued. Polling operation status...")
      while not operation.done:
        time.sleep(10)
        operation = self._client.operations.get(operation)
        logging.info(f"Veo Polling: Operation status done={operation.done}")

      self._time_completed = time.time()
      self.log(f"Video generation success with {self._model_name}")

      if operation.response and operation.response.generated_videos:
        generated_video = operation.response.generated_videos[0]
        response = {"video": generated_video, "client": self._client}
        return callback(prompt, response)
      else:
        raise RuntimeError("Veo generation completed, but returned no generated video data.")

    except Exception as e:
      self._time_completed = time.time()
      self.log(f"Video generation error with {self._model_name}", is_error=True)
      self.log(str(e), is_error=True)
      return callback(prompt, {"error": str(e)})
