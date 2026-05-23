from collections.abc import Callable
import logging

from pydantic import BaseModel
import time
import threading

from .gemini import Gemini, GEMINI_MODEL_NAME_FAST, NANO_BANANA_MODEL_NAME_PRO


LLM_TYPES = {"gemini": [GEMINI_MODEL_NAME_FAST, NANO_BANANA_MODEL_NAME_PRO]}


class LLM:

  def __init__(self, model_type: str, embedding_model_name: str = None):
    assert model_type in LLM_TYPES
    self._model_type = model_type
    model_name = LLM_TYPES[model_type]
    if model_type == "gemini":
      self._llm = Gemini(model_name[0], model_name[1])
      self._not_blocking = True
    else:
      self._llm = None
      self._not_blocking = False

    self._stack = []
    self._busy = False


  def reset(self) -> bool:
    self._busy = False


  @property
  def type(self) -> str:
    return self._model_type


  @property
  def name(self) -> str:
    return str(LLM_TYPES[self._model_type])


  def embed(self,
            text: str,
            callback: Callable = None,
            callback_failed: Callable = None):
    """Embed a `text` string using an LLM. Results will be returned via `callback`.

    Args:
      text: (str) Text string for the LLM.
      callback: (abc.Callable) Function that will be called once the embedding finished. Default None.
      callback_failed: (abc.Callable) Function that will be called if embedding failed. Default None.
    """

    logging.debug("Running %s for embedding text id: %s (blocking? %s)",
        self._model_type, str(not self._not_blocking))
    t0 = time.time()
    self._llm.embed(text, callback, callback_failed)
    time_embed = time.time() - t0
    logging.debug("Embedding total %.2fs", time_embed)


  def generate(self,
               prompt: str,
               callback: Callable = None,
               response_format: BaseModel = None,
               thinking: bool = False,
               max_tokens: int = 256,
               is_image: bool = False):
    """Generate an LLM response to a `prompt`. Results will be returned via `callback`.

    Args:
      prompt: (str) Text prompt for the LLM.
      callback: (abc.Callable) Function that will be called once the generation finished. Default None.
      response_format: (pydantic.BaseModel) Class with response format. Default None.
      thinking: (bool) Whether thinking is on for the LLM call.
      max_tokens: (int) Number of tokens to generate.
      is_image: (bool) Whether to use the image generation model. Default False.
    """

    logging.debug("Running %s for prompt id: %s (blocking? %s)",
        self._model_type, prompt, str(not self._not_blocking))
    t0 = time.time()
    self._llm.generate(prompt, callback, response_format, thinking, max_tokens,
                       is_image=is_image)
    time_gen = time.time() - t0
    logging.debug("Generation total %.2fs", time_gen)

    # Process next prompts?
    self._busy = False
    self._next()


  def schedule_generate(self,
                        prompt: str,
                        callback: Callable = None,
                        response_format: BaseModel = None,
                        thinking: bool = False,
                        max_tokens: int = 256,
                        is_image: bool = False):
    """Schedule a generation by adding its parameters to a stack.

    This function allows to generate multiple LLM calls one after the other.

    Args:
      prompt: (str) Text prompt for the LLM.
      callback: (abc.Callable) Function that will be called once the generation finished. Default None.
      response_format: (pydantic.BaseModel) Class with response format. Default None.
      thinking: (bool) Whether thinking is on for the LLM call.
      max_tokens: (int) Number of tokens to generate.
      is_image: (bool) Whether to use the image generation model. Default False.
    """

    logging.debug("Adding to stack: %s", prompt)
    self._stack.append((prompt, callback, response_format, thinking, max_tokens,
                        is_image))
    if self._busy is False or self._not_blocking:
      logging.debug("Scheduling immediately: %s", prompt)
      self._next()


  def _next(self):
    """Process the next element on the stack."""

    if len(self._stack) > 0:
      self._busy = True
      (prompt, callback, response_format, thinking, max_tokens,
       is_image) = self._stack.pop()
      thread = threading.Thread(
          target=self.generate,
          args=(prompt, callback, response_format, thinking, max_tokens,
                is_image))
      thread.start()
    else:
      logging.info("Processing stack emptied")
