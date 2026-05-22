import io
from typing import Optional

import cv2
from cv2_enumerate_cameras import enumerate_cameras
import numpy as np
from PIL import Image
from PIL import ImageOps


def get_image_from_bytes(response_content: bytes) -> Image.Image:
  """
  Convert raw bytes into a PIL Image and ensure it is in RGB format.

  Args:
    response_content (bytes): The raw image byte data.

  Returns:
    Image.Image: The resulting RGB PIL Image.
  """
  img = Image.open(io.BytesIO(response_content))
  if img.mode != 'RGB':
    img = img.convert('RGB')
  return img


def resize_image(img: Image.Image, width: int, height: int) -> Image.Image:
  """
  Resize and pad a PIL Image to the target dimensions while preserving its aspect ratio.

  Args:
    img (Image.Image): The original PIL Image.
    width (int): Target width in pixels.
    height (int): Target height in pixels.

  Returns:
    Image.Image: The resized and padded PIL Image.
  """
  try:
    resample = Image.Resampling.LANCZOS
  except AttributeError:
    resample = Image.LANCZOS
  img = ImageOps.pad(img, (width, height), method=resample, color=(0, 0, 0))
  return img


def list_webcams(device_name: str) -> int:
  """
  Enumerate all connected webcams and find the first matching the given device name.

  Args:
    device_name (str): The substring to search for in the camera names.

  Returns:
    int: The index of the first matched camera, or 0 if none match.
  """
  ids = []
  for camera_info in enumerate_cameras():
    print(f"{camera_info.index}: {camera_info.name}")
    if device_name in camera_info.name:
      ids.append(camera_info.index)
  return min(ids) if ids else 0


def write_numpy_to_byte_string(image: np.ndarray) -> Optional[bytes]:
  """
  Convert a NumPy BGR image array into a JPEG byte stream.

  Args:
    image (np.ndarray): The input image array in BGR format (e.g., from OpenCV).

  Returns:
    Optional[bytes]: The JPEG encoded bytes, or None if the input image is None.
  """
  if image is not None:
    frame = io.BytesIO()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image.save(frame, format='JPEG')
    return frame.getvalue()
  else:
    return None


def log(msg: str, msg_type: str) -> None:
  print(f"[{msg_type}] {msg}")
