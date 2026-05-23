import logging


from .utils import COLOR_ERROR, COLOR_LLM, COLOR_RESET


class GenericLLM:
  """Class interfacing with Ollama architectures."""

  def __init__(self):
    self.reset()


  def reset(self):
    """Reset the timers and the LLM busy state."""
    self._time_completed = 0
    self._time_scheduled = 0


  @property
  def time_completed(self):
    return self._time_completed
  

  @property
  def time_scheduled(self):
    return self._time_scheduled


  def busy(self):
    """Is the system busy?"""
    busy = self._time_completed < self._time_scheduled
    return busy


  def log(self, response: str, is_error: bool = False):
    if is_error:
      logging.warning("[%.2fs] %s%s%s", self._time_completed-self._time_scheduled,
                      COLOR_ERROR, response, COLOR_RESET)
    else:
      logging.info("[%.2fs] %s%s%s", self._time_completed-self._time_scheduled,
                   COLOR_LLM, response, COLOR_RESET)
