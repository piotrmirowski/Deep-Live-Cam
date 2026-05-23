from collections.abc import Callable
import logging

import json
import os
from pydantic import BaseModel
import time

from google import genai
from google.genai import types

from .generic_llm import GenericLLM
from .utils import split_prompt_into_system_user


GEMINI_MODEL_NAME_FAST = "gemini-3-flash-preview"
GEMINI_MODEL_NAME_PRO = "gemini-3.1-pro-preview"
NANO_BANANA_MODEL_NAME_FAST = "gemini-3.1-flash-image-preview"
NANO_BANANA_MODEL_NAME_PRO = "gemini-3-pro-image-preview"
GEMINI_EMBEDDING_MODEL_NAME = "gemini-embedding-exp-03-07"
GEMINI_TEMPERATURE = 1.5
GEMINI_TOP_K = 1
GEMINI_TOP_P = 0.9
GEMINI_MAX_TOKENS = 256
GEMINI_THINKING_LEVEL_FAST = "minimal"
GEMINI_THINKING_LEVEL_PRO = "low"


class Gemini(GenericLLM):
  """Class interfacing with Google's Gemini-based architectures."""

  def __init__(self,
               model_name: str = GEMINI_MODEL_NAME_FAST,
               image_model_name: str = NANO_BANANA_MODEL_NAME_FAST,
               embedding_model_name: str = GEMINI_EMBEDDING_MODEL_NAME):
    super().__init__()
    api_key = os.getenv("GEMINI_API_KEY")
    assert api_key, "Please set GEMINI_API_KEY environment variable"
    self._model_name = model_name
    self._image_model_name = image_model_name
    self._embedding_model_name = embedding_model_name
    self._client = genai.Client(api_key=api_key)
    logging.info("Configured Gemini with GEMINI_API_KEY=%s", api_key)


  def generate(self,
               prompt: str = "",
               callback: Callable = None,
               response_format: BaseModel = None,
               thinking: bool = False,
               max_tokens: int = GEMINI_MAX_TOKENS,
               is_image: bool = False):
    """Generate a Gemini response to a `prompt`. Results will be returned via `callback`.

    Args:
      prompt: (str) Text prompt for the LLM.
      callback: (abc.Callable) Function that will be called once the generation finished. Default None.
      response_format: (pydantic.BaseModel) Class with response format. Default None.
      thinking: (bool) Whether thinking is on for the LLM call.
      max_tokens: (int) Number of tokens to generate.
      is_image: (bool) Whether to use Nano Banana image generation. Default False.
    """

    # Route to image generation when requested.
    if is_image:
      return self.generate_image(prompt, callback, response_format=response_format)

    system_prompt, user_prompt = split_prompt_into_system_user(prompt)
    config = types.GenerateContentConfig()
    config.system_instruction = system_prompt
    config.temperature = GEMINI_TEMPERATURE
    config.top_k = GEMINI_TOP_K
    config.top_p = GEMINI_TOP_P

    # We ignore max_tokens and let the model choose
    # config.max_output_tokens = max_tokens
    if thinking:
      config.thinking_config = types.ThinkingConfig(
          thinking_level=GEMINI_THINKING_LEVEL_PRO,
          include_thoughts=True)
      is_thinking = "slow|"
    else:
      config.thinking_config = types.ThinkingConfig(
          thinking_level=GEMINI_THINKING_LEVEL_FAST,
          include_thoughts=True)
      is_thinking = "fast|"
    if response_format is not None:
      config.response_mime_type = "application/json"
      config.response_schema = response_format
    try:
      self._time_scheduled = time.time()
      response_obj = self._client.models.generate_content(
          contents=user_prompt, model=self._model_name, config=config)
      response = response_obj.text
      self._time_completed = time.time()
      self.log(is_thinking + "\n" + response)
      if response_format is not None:
        response = json.loads(response)
      else:
        response = response.strip()
      self.log(f"Success in {'slow' if thinking else 'fast'} mode")
      return callback(prompt, response)
    except Exception as e:
      self.log(f"Error in {'slow' if thinking else 'fast'} mode", is_error=True)
      self.log(e, is_error=True)
      self.log(response_obj.text, is_error=True)
      return callback(prompt, {"error": str(e)})


  def generate_image(self,
                     prompt: str = "",
                     callback: Callable = None,
                     response_format: dict[str, str] = {}):
    """Generate an image using the Nano Banana model. Results will be returned via `callback`.

    Args:
      prompt: (str) Text prompt describing the image to generate.
      callback: (abc.Callable) Function called with (prompt, response_obj) on completion.
    """
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio=response_format["aspect_ratio"],
            image_size=response_format["image_size"]))
    try:
      self._time_scheduled = time.time()
      response_obj = self._client.models.generate_content(
          contents=prompt, model=self._image_model_name, config=config)
      self._time_completed = time.time()
      self.log(f"Image generation success with {self._image_model_name}")
      response = {"text": None, "image": None}
      for part in response_obj.parts:
        if part.text is not None:
          response["text"] = part.text
        elif part.inline_data is not None:
          response["image"] = part.as_image()
      return callback(prompt, response)
    except Exception as e:
      self.log(f"Image generation error with {self._image_model_name}",
              is_error=True)
      self.log(e, is_error=True)
      return callback(prompt, {"error": str(e)})


  def embed(self,
            text: str = "",
            callback: Callable = None,
            callback_failed: Callable = None):
    """Embed a `text` string using Gemini. Results will be returned via `callback`.

    Args:
      text: (str) Text string for the LLM.
      callback: (abc.Callable) Function that will be called once the embedding finished. Default None.
      callback_failed: (abc.Callable) Function that will be called if embedding failed. Default None.
    """

    try:
      response_obj = self._client.models.embed_content(
          contents=text,
          model=self._embedding_model_name,
          config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"))
      response = response_obj.embeddings[0].values
      return callback(text, response)
    except Exception as e:
      logging.warning(e)
      return callback_failed(text, None)
