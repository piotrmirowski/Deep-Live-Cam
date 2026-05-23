COLOR_LLM = "\x1b[33;20m"
COLOR_ERROR = "\x1b[0;31m"
COLOR_RESET = "\x1b[0m"


def split_prompt_into_system_user(prompt: str) -> list[str]:
  """Split a text `prompt` into a pair of `system_prompt` and `user_prompt`.

  Args:
    prompt: (str) Prompt.
  Returns:
    system_prompt, user_prompt: (list[str]) System and user prompts.
  """

  lines = prompt.split('\n\n')
  system_prompt = '\n\n'.join(lines[:1])
  user_prompt = '\n\n'.join(lines[1:])
  return system_prompt, user_prompt


def format_name(name: str):
  """Format the speaker `name` as **name:**.

  Args:
    name: (str) Name.
  Returns:
    name: (str) Formatted name.
  """

  return f"**{name}:** "


def remove_quotes(text: str) -> str:
  """Clean the `text` from extra quotes at start and end of string.

  Args:
    text: (str) Text with quotes to be removed.
  Returns:
    text: (str) Text without extra quotes.
  """

  if len(text) < 2:
    return text
  if (text.startswith('"') and text.endswith('"')) or (
      text.startswith("'") and text.endswith("'")):
    text = text[1:-1].strip()

  def _count(char_to_find, s):
    return len([ch for ch in s if ch == char_to_find])

  if text.startswith('“') and _count('“', text) == 1:
    text = text[1:]
  if text.startswith('"') and _count('"', text) == 1:
    text = text[1:]
  if text.startswith("'") and _count("'", text) == 1:
    text = text[1:]
  if text.endswith('”') and _count('”', text) == 1:
    text = text[:-1]
  if text.endswith('"') and _count('"', text) == 1:
    text = text[:-1]
  if text.endswith("'") and _count("'", text) == 1:
    text = text[:-1]

  return text
