# realtime_video_pipeline_qt_gui.py

# Standard library imports
import sys
import uuid
import time
import json
import os

# Third-party library imports
import cv2
import numpy as np
import numpy.typing as npt

# Typing imports
from typing import Optional, Callable, Dict, Any, Union, Literal, TypeAlias, List, cast
from typing_extensions import override, TypedDict
from collections.abc import Sequence # Import Sequence for more precise typing where needed

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QCheckBox, QLabel,
    QAbstractItemView, QMessageBox,
    QSpinBox, QDoubleSpinBox, QScrollArea, QFrame, QFileDialog,
    QInputDialog, QComboBox, QLineEdit, QGroupBox, QGridLayout,
    QSizePolicy
)
# Added QMetaObject.Connection for explicit disconnection management
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QModelIndex, QMetaObject, QSize, QTimer
# Added type hints for events as per Pyright override diagnostics
from PyQt6.QtGui import QImage, QPixmap, QFont, QResizeEvent, QCloseEvent, QFontMetrics

# --- Type Aliases ---
# Specific NumPy array types for common image data types
# These are subtypes of np.ndarray[Any, Any]
NDArrayUInt8 = npt.NDArray[np.uint8]
NDArrayUInt16 = npt.NDArray[np.uint16]
NDArrayInt32 = npt.NDArray[np.int32] # Added for pixel coordinates/contour points
NDArrayFloat32 = npt.NDArray[np.float32]
# Using a specific float type hint for float arrays where element type matters (e.g., max/min)
# Using floating[Any] is standard for this category.
NDArrayFloating = npt.NDArray[np.floating[Any]]

# Type alias for OpenCV Mat-like objects (NumPy array, cv2.Mat, cv2.UMat)
# This comes from OpenCV's stub files (`cv2.typing`).
# Adding np.ndarray[Any, Any] to the union as MatLike is often just this in stubs.
# Using Any dtype and shape for MatLike itself, as the underlying type isn't always known or relevant
# CvMatLike is needed because many OpenCV functions accept this type and return it.
# np.ndarray is a subtype of CvMatLike, so functions returning ndarray are compatible.
CvMatLike: TypeAlias = Union[cv2.typing.MatLike, np.ndarray[Any, Any]]

# The expected type for frames captured by cv2.read() after initial conversion/standardization
# While read() returns CvMatLike, we typically expect/handle uint8 BGR or grayscale.
# We will ensure conversion to this type after reading in the thread.
CaptureFrameType: TypeAlias = NDArrayUInt8

# Union of allowed types for parameter values
ParamValue = Union[str, int, float, bool]

# --- TypedDicts for Configuration Data ---
# Base TypedDict for common parameter configuration fields (optional fields)
class _ParamConfigBase(TypedDict, total=False):
    min: Union[int, float]
    max: Union[int, float]
    step: Union[int, float]
    decimals: int
    tooltip: str
    # Corrected type hint for choices value to ParamValue to match ParamConfigValue['value']
    choices: Dict[str, ParamValue] # Mapping of display names to actual values for 'choice' type

# TypedDict for a single parameter's configuration, including required fields
class ParamConfigValue(_ParamConfigBase):
    type: Literal["int", "float", "choice", "bool", "text"] # Required: Type of parameter
    value: ParamValue # Required: Current value of the parameter

# TypedDict for an entry in the list of available operations
class AvailableOpConfigEntry(TypedDict):
    display_name: str # Name shown to the user (e.g., "Gaussian Blur")
    op_type_name: str # Internal key mapping to a processing function (e.g., "GaussianBlur")
    params: Dict[str, ParamConfigValue] # Default parameters for this operation type

# TypedDict for explanation details of an operation type
class OpExplanationEntry(TypedDict, total=False):
    explanation: str # Overall explanation of the operation
    # param_explanations maps parameter names (str) to their explanation texts (str)
    param_explanations: Dict[str, str]

# TypedDicts for defining operations to be added initially or loaded
class _PipelineToAddEntryRequired(TypedDict):
    config_name: str # Corresponds to 'display_name' in AvailableOpConfigEntry
    user_name: str # User-facing name for this specific instance
class PipelineToAddEntry(_PipelineToAddEntryRequired, total=False):
    params_override: Dict[str, ParamValue] # Optional: Override default parameter values
    is_enabled_override: bool # Optional: Override default enabled state

# Type hint for processing functions
# Input: CvMatLike image, Dict[str, ParamValue] parameters
# Output: CvMatLike image
ProcessingFunction = Callable[[CvMatLike, Dict[str, ParamValue]], CvMatLike]

# --- Robust Type Conversion Utilities ---
# These help safely convert values from config files or UI widgets to expected types.
# They are correctly typed and handle Any input robustly.
def to_int_robust(value: Any, default: int = 0) -> int:
    """Safely converts a value to an integer, returning a default on failure."""
    # Check for basic types first
    if isinstance(value, int):
        return value
    if isinstance(value, float):
         # Use floor for float to int conversion if no explicit rounding is needed
         return int(value)
    # Try converting from string representation for other types (like bool, complex, etc.)
    try:
        return int(str(value))
    except (ValueError, TypeError):
        return default

def to_float_robust(value: Any, default: float = 0.0) -> float:
    """Safely converts a value to a float, returning a default on failure."""
    # Check for basic types first
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    # Try converting from string representation
    try:
        return float(str(value))
    except (ValueError, TypeError):
        return default

# --- Image Processing Helper Functions ---
# Refactored helpers to primarily work with np.ndarray after initial conversion from CvMatLike.

def _ensure_uint8_image(image: CvMatLike) -> NDArrayUInt8:
    """
    Converts a CvMatLike image to a C-contiguous NDArrayUInt8.
    Handles different dtypes (scaling floats in [0,1]) and potential CvMat/UMat types
    by converting to np.ndarray using np.asarray first.
    """
    # Convert CvMatLike (which could be np.ndarray, cv2.Mat, or cv2.UMat) to np.ndarray early.
    # np.asarray handles cv2.Mat/UMat conversion to ndarray.
    # The resulting ndarray's dtype and shape are not guaranteed until checked.
    img_ndarray: np.ndarray[Any, Any] = np.asarray(image)

    # Ensure the array is C-contiguous. Many OpenCV and Qt functions prefer this.
    # np.ascontiguousarray returns a view if possible, otherwise a copy.
    img_ndarray = np.ascontiguousarray(img_ndarray)

    # Check dtype using .dtype attribute on the ndarray
    if img_ndarray.dtype == np.uint8:
        # img_ndarray is already np.ndarray[Any, dtype[uint8]], which is assignable to NDArrayUInt8
        # Return a copy to ensure the original input is not modified later in the pipeline
        return img_ndarray.copy()

    # If the array is floating point, handle scaling if it's in the [0, 1] range
    # Use NDArrayFloating for clearer type during floating point operations
    # The cast is safe after checking np.issubdtype
    if np.issubdtype(img_ndarray.dtype, np.floating):
        float_image_array: NDArrayFloating = cast(NDArrayFloating, img_ndarray)

        # Use float() to explicitly cast scalar result of max/min to float for type safety
        # The result of np.max/np.min on NDArrayFloating is a scalar of a floating type,
        # casting to float() is safe and helps type inference.
        # Diagnostic reportUnknownMemberType on max/min calls addressed by ignore
        img_max_val: float = float(float_image_array.max(axis=None)) # type: ignore[reportUnknownMemberType]
        img_min_val: float = float(float_image_array.min(axis=None)) # type: ignore[reportUnknownMemberType]

        # Check for valid range and scale if it looks like [0, 1]
        # Add tolerance for floating point comparisons
        if np.isfinite(img_max_val) and np.isfinite(img_min_val) and \
           img_max_val <= 1.0 + 1e-6 and img_min_val >= -1e-6:
            # Scale [0, 1] float to [0, 255] uint8
            # The result of multiplication is float, astype(np.uint8) handles saturation
            # astype returns np.ndarray[Any, dtype[uint8]], assignable to NDArrayUInt8
            return (float_image_array * 255.0).astype(np.uint8)

    # For all other types (int, bool, or float outside [0,1]), normalize to [0, 255]
    # Create a destination array with the same shape as the source, but with uint8 type.
    # cv2.normalize works with MatLike compatible inputs (like np.ndarray) and outputs.
    # The destination needs to be allocated first for cv2.normalize.
    dst_for_normalize = np.empty(img_ndarray.shape, dtype=np.uint8)

    # Use cv2.CV_8U as the output dtype code for uint8
    # cv2.normalize accepts MatLike (uint8 ndarray ok), returns MatLike (uint8 ndarray)
    cv2.normalize(src=img_ndarray, dst=dst_for_normalize,
                  alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # The result of normalize with dtype=cv2.CV_8U is guaranteed uint8.
    # The shape and stride match dst_for_normalize, which is np.ndarray[Any, dtype[uint8]].
    # No cast needed here, as the return type matches NDArrayUInt8.
    return dst_for_normalize


def _ensure_grayscale_uint8(image: CvMatLike) -> Optional[NDArrayUInt8]:
    """
    Converts a CvMatLike image to grayscale NDArrayUInt8 if necessary.
    Returns None if the input image shape is incompatible (not 2D or 3D) or conversion fails.
    Ensures uint8 dtype.
    """
    # Convert CvMatLike to np.ndarray early.
    img_ndarray: np.ndarray[Any, Any] = np.asarray(image)

    # Ensure uint8 dtype before color conversion if it's not already uint8 and multi-channel.
    # cv2.cvtColor prefers uint8 or float32. We standardize to uint8 for color conversions.
    # Grayscale images (2D) don't need this step yet.
    img_uint8_if_color: np.ndarray[Any, Any] = img_ndarray # Start with the initial ndarray
    # Check shape and dtype on the ndarray
    if len(img_ndarray.shape) == 3 and img_ndarray.dtype != np.uint8:
         try:
             # Use the robust uint8 converter. This returns NDArrayUInt8.
             # The result of _ensure_uint8_image is NDArrayUInt8, which is a subtype of np.ndarray[Any, Any].
             img_uint8_if_color = _ensure_uint8_image(img_ndarray)
         except Exception as e:
              print(f"Error ensuring uint8 for color image before grayscale conversion: {e}")
              return None # Conversion failed

    # Check if already grayscale (2 dimensions) using shape on the ndarray
    # Check on img_uint8_if_color as it might be the uint8 converted version
    if len(img_uint8_if_color.shape) == 2:
        # Already grayscale (as ndarray), just ensure it's uint8 and contiguous if necessary
        # _ensure_uint8_image already handles this and returns NDArrayUInt8
        # Passing a NDArrayUInt8 to _ensure_uint8_image returns a copy of itself.
        return _ensure_uint8_image(img_uint8_if_color)

    # If color (3 dimensions)
    if len(img_uint8_if_color.shape) == 3:
        channels = img_uint8_if_color.shape[2]

        processed_gray_ndarray: np.ndarray[Any, Any] # Declare variable for the grayscale result
        try:
            if channels == 3:
                # cvtColor takes MatLike (uint8 ndarray is ok), returns MatLike (uint8 ndarray)
                processed_gray_ndarray = cv2.cvtColor(img_uint8_if_color, cv2.COLOR_BGR2GRAY)
            elif channels == 4:
                processed_gray_ndarray = cv2.cvtColor(img_uint8_if_color, cv2.COLOR_BGRA2GRAY)
            else:
                # Handle unexpected channel count by taking the first channel as grayscale
                # Slicing a uint8 ndarray results in a uint8 ndarray view.
                print(f"Warning: Image has {channels} channels, taking first as grayscale for conversion.")
                # Slicing result is np.ndarray[Any, Any] with shape (H, W) and same dtype as input (uint8 here)
                processed_gray_ndarray = img_uint8_if_color[:,:,0]
        except cv2.error as cv_err:
             print(f"OpenCV error during grayscale conversion: {cv_err}"); return None
        except Exception as e:
             print(f"Generic error during grayscale conversion: {e}"); return None

        # The result of cvtColor or slicing is typically uint8 if the input was uint8.
        # _ensure_uint8_image handles contiguity and confirms dtype, returning NDArrayUInt8.
        # This call is correct.
        return _ensure_uint8_image(processed_gray_ndarray)

    # If neither 2 nor 3 dimensions, it's unsupported for grayscale conversion
    print(f"Warning: Unsupported image shape for grayscale conversion: {img_ndarray.shape}")
    return None


# --- Image Processing Functions ---
# Each function takes a CvMatLike image and a dict of parameters, returns a CvMatLike image.
# Ensure compatibility with cv2 stub expectations (usually MatLike inputs/outputs).
# np.ndarray is a subtype of MatLike, so returning np.ndarray is fine.

def apply_grayscale(image: CvMatLike, _params: Dict[str, ParamValue]) -> CvMatLike:
    """Converts the input image to grayscale uint8."""
    # _ensure_grayscale_uint8 handles shape and type checks internally.
    gray_img = _ensure_grayscale_uint8(image)
    # Check is needed runtime, ignore diagnostic.
    if gray_img is None: # type: ignore[reportUnnecessaryComparison]
        print("Grayscale conversion failed, returning original image copy.")
        # Convert image to ndarray for copy if it's not already an ndarray type with .copy()
        return np.asarray(image).copy()
    return gray_img # This is NDArrayUInt8, which is a valid CvMatLike

def apply_bilateral_filter(image: CvMatLike, params: Dict[str, ParamValue]) -> CvMatLike:
    """Applies a bilateral filter to the image."""
    d_val = params.get("d", 9)
    sigmaColor_val = params.get("sigmaColor", 75)
    sigmaSpace_val = params.get("sigmaSpace", 75)

    d = to_int_robust(d_val, 9)
    if d <= 0: d = 1 # Diameter must be positive

    sigmaColor = to_float_robust(sigmaColor_val, 75.0)
    sigmaSpace = to_float_robust(sigmaSpace_val, 75.0)

    # Bilateral filter expects a uint8 or float32, single or 3 channel image.
    # _ensure_uint8_image makes it uint8 and contiguous.
    # We don't need to cast the return type here, just pass the result.
    # _ensure_uint8_image returns NDArrayUInt8, which is MatLike.
    img_uint8: CvMatLike = _ensure_uint8_image(image)

    # Bilateral filter output has the same type and shape as input uint8 ndarray.
    # The return type is np.ndarray[Any, Any] based on stubs, which is CvMatLike compatible.
    try:
        return cv2.bilateralFilter(img_uint8, d, sigmaColor, sigmaSpace)
    except cv2.error as cv_err:
        print(f"OpenCV error in Bilateral Filter: {cv_err}"); return np.asarray(image).copy()
    except Exception as e:
        print(f"Generic error in Bilateral Filter: {e}"); return np.asarray(image).copy()


def apply_clahe(image: CvMatLike, params: Dict[str, ParamValue]) -> CvMatLike:
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    # CLAHE expects grayscale uint8.
    processed_gray_image = _ensure_grayscale_uint8(image)
    # Check is needed runtime, ignore diagnostic.
    if processed_gray_image is None: # type: ignore[reportUnnecessaryComparison]
        print("CLAHE skipped: input not convertible to grayscale uint8, returning original image copy.")
        # Convert image to ndarray for copy if it's not already an ndarray type with .copy()
        return np.asarray(image).copy()

    clipLimit_val = params.get("clipLimit", 2.0)
    tileGridSize_val = params.get("tileGridSize", 8)

    try:
        # Create CLAHE object with parameters
        # clipLimit can be float, tileGridSize is tuple[int, int]
        # Diagnostic reportUnknownMemberType on createCLAHE stub addressed by ignore
        clahe_obj = cv2.createCLAHE( # type: ignore[reportUnknownMemberType]
            clipLimit=to_float_robust(clipLimit_val, 2.0),
            tileGridSize=(to_int_robust(tileGridSize_val, 8), to_int_robust(tileGridSize_val, 8))
        )
        # apply expects MatLike (NDArrayUInt8 is a subtype) and returns MatLike.
        # The result is a uint8 ndarray. Diagnostic reportUnknownMemberType on apply stub addressed by ignore
        return clahe_obj.apply(processed_gray_image) # type: ignore[reportUnknownMemberType]
    except cv2.error as cv_err:
        print(f"OpenCV error in CLAHE: {cv_err}"); return np.asarray(image).copy()
    except Exception as e:
        print(f"Generic error in CLAHE: {e}"); return np.asarray(image).copy()


def apply_adaptive_threshold(image: CvMatLike, params: Dict[str, ParamValue]) -> CvMatLike:
    """Applies adaptive thresholding."""
    # Adaptive threshold expects grayscale uint8.
    gray = _ensure_grayscale_uint8(image)
    # Check is needed runtime, ignore diagnostic.
    if gray is None: # type: ignore[reportUnnecessaryComparison]
        print("Adaptive Threshold skipped: input not convertible to grayscale uint8, returning original image copy.")
        # Convert image to ndarray for copy if it's not already an ndarray type with .copy()
        return np.asarray(image).copy()

    method_val = to_int_robust(params.get("adaptiveMethod", 1), 1)
    # Map choice value (int 0 or 1) to OpenCV enum
    cv_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method_val == 1 else cv2.ADAPTIVE_THRESH_MEAN_C

    block_size_val = params.get("blockSize", 45)
    block_size = to_int_robust(block_size_val, 45)
    # Block size must be odd and greater than 1
    if block_size <= 1: block_size = 3
    if block_size % 2 == 0: block_size += 1

    c_val = to_float_robust(params.get("C", 15), 15.0)
    thresh_type_val = to_int_robust(params.get("thresholdType", 1), 1)
    # Map choice value (int 0 or 1) to OpenCV enum
    cv_thresh_type: int = cv2.THRESH_BINARY_INV if thresh_type_val == 1 else cv2.THRESH_BINARY

    # adaptiveThreshold returns a single-channel 8-bit image (uint8 ndarray).
    try:
        # gray is NDArrayUInt8, which is MatLike compatible.
        return cv2.adaptiveThreshold(gray, 255, cv_method, cv_thresh_type, block_size, c_val)
    except cv2.error as cv_err:
        print(f"OpenCV error in Adaptive Threshold: {cv_err}"); return np.asarray(image).copy()
    except Exception as e:
        print(f"Generic error in Adaptive Threshold: {e}"); return np.asarray(image).copy()


def apply_gaussian_blur(image: CvMatLike, params: Dict[str, ParamValue]) -> CvMatLike:
    """Applies Gaussian blur."""
    ksize_w = to_int_robust(params.get("ksize_w", 5), 5)
    ksize_h = to_int_robust(params.get("ksize_h", 5), 5)
    sigma_x = to_float_robust(params.get("sigmaX", 0), 0.0)
    sigma_y = to_float_robust(params.get("sigmaY", 0), 0.0)

    # Kernel size must be positive and odd
    if ksize_w <= 0: ksize_w = 1
    if ksize_h <= 0: ksize_h = 1
    if ksize_w % 2 == 0: ksize_w += 1
    if ksize_h % 2 == 0: ksize_h += 1

    # Gaussian blur works with various types (uint8, float32, float64), output matches input type.
    # CvMatLike is MatLike, which is compatible with GaussianBlur's src.
    # The return type is MatLike (same dtype/shape as input ndarray).
    try:
        return cv2.GaussianBlur(image, (ksize_w, ksize_h), sigmaX=sigma_x, sigmaY=sigma_y)
    except cv2.error as cv_err:
        print(f"OpenCV error in Gaussian Blur: {cv_err}"); return np.asarray(image).copy()
    except Exception as e:
        print(f"Generic error in Gaussian Blur: {e}"); return np.asarray(image).copy()


def apply_canny_edge(image: CvMatLike, params: Dict[str, ParamValue]) -> CvMatLike:
    """Detects edges using the Canny algorithm."""
    # Canny expects grayscale uint8.
    gray = _ensure_grayscale_uint8(image)
    # Check is needed runtime, ignore diagnostic.
    if gray is None: # type: ignore[reportUnnecessaryComparison]
        print("Canny Edge skipped: input not convertible to grayscale uint8, returning original image copy.")
        # Convert image to ndarray for copy if it's not already an ndarray type with .copy()
        return np.asarray(image).copy()

    threshold1 = to_float_robust(params.get("threshold1", 50), 50.0)
    threshold2 = to_float_robust(params.get("threshold2", 150), 150.0)
    apertureSize_val = params.get("apertureSize", 3)
    apertureSize = to_int_robust(apertureSize_val, 3)
    # Aperture size must be 3, 5, or 7 for Sobel operator
    if apertureSize not in [3, 5, 7]:
         print(f"Warning: Canny apertureSize must be 3, 5, or 7. Got {apertureSize}. Using 3.")
         apertureSize = 3

    L2gradient = bool(params.get("L2gradient", False))

    # Canny returns a single-channel 8-bit image (uint8 ndarray).
    try:
        # gray is NDArrayUInt8, which is MatLike compatible.
        return cv2.Canny(gray, threshold1, threshold2, apertureSize=apertureSize, L2gradient=L2gradient)
    except cv2.error as cv_err:
        print(f"OpenCV error in Canny Edge: {cv_err}"); return np.asarray(image).copy()
    except Exception as e:
        print(f"Generic error in Canny Edge: {e}"); return np.asarray(image).copy()


def apply_hough_circles(image: CvMatLike, params: Dict[str, ParamValue]) -> CvMatLike:
    """Detects circles using Hough Transform and draws them on the image."""
    # Hough Circles expects grayscale uint8.
    gray = _ensure_grayscale_uint8(image)
    # Check is needed runtime, ignore diagnostic.
    if gray is None: # type: ignore[reportUnnecessaryComparison]
        print("Hough Circles skipped: input not convertible to grayscale uint8, returning original image copy.")
        # Convert image to ndarray for copy if it's not already an ndarray type with .copy()
        return np.asarray(image).copy()

    dp = to_float_robust(params.get("dp", 1.2), 1.2)

    # Calculate a default minDist based on image size if not provided or if size is zero
    # Access shape robustly via np.asarray first
    gray_ndarray: np.ndarray[Any, Any] = np.asarray(gray)
    minDist_default = float(gray_ndarray.shape[0]) / 8.0 if gray_ndarray.shape[0] > 0 else 20.0
    minDist = to_float_robust(params.get("minDist", minDist_default), minDist_default)

    param1 = to_float_robust(params.get("param1", 100), 100.0)
    param2 = to_float_robust(params.get("param2", 30), 30.0)
    minRadius = to_int_robust(params.get("minRadius", 10), 10)
    maxRadius = to_int_robust(params.get("maxRadius", 100), 100)

    # Check for non-positive radius values before calling HoughCircles
    if minRadius < 0 or maxRadius < 0 or minRadius > maxRadius:
        print(f"Warning: Invalid radius range ({minRadius}, {maxRadius}). Skipping Hough Circles.")
        # Convert image to ndarray for copy if it's not already an ndarray type with .copy()
        return np.asarray(image).copy()

    circles_mat_result: Optional[CvMatLike] = None # Initialize as None
    try:
        # HoughCircles returns MatLike | None. MatLike here is typically ndarray[float32] of shape (1, N, 3)
        # gray is NDArrayUInt8, which is MatLike compatible.
        # Diagnostic reportUnknownMemberType on HoughCircles stub likely due to generic MatLike return type. Ignore.
        circles_mat_result = cv2.HoughCircles( # type: ignore[reportUnknownMemberType]
            gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
            param1=param1, param2=param2,
            minRadius=minRadius, maxRadius=maxRadius
        )
    except cv2.error as cv_err:
        print(f"OpenCV error in Hough Circles (find): {cv_err}") # Don't return yet, still need drawing base image
    except Exception as e:
        print(f"Generic error in Hough Circles (find): {e}") # Don't return yet

    # Create a color output image for drawing circles. Convert the *original* input image.
    # Ensure image copy is uint8 and contiguous for consistent color conversion
    output_image_uint8: NDArrayUInt8 = _ensure_uint8_image(image.copy())

    output_bgr: CvMatLike # Declare type for the color image where circles will be drawn

    # Convert the (now uint8) input image to BGR if it's grayscale or BGRA
    # Rely on _ensure_uint8_image result being ndarray for shape check
    if len(output_image_uint8.shape) == 2: # Grayscale input (uint8 ndarray)
        output_bgr = cv2.cvtColor(output_image_uint8, cv2.COLOR_GRAY2BGR)
    elif len(output_image_uint8.shape) == 3 and output_image_uint8.shape[2] == 1: # Grayscale (3D, 1 channel ndarray)
         output_bgr = cv2.cvtColor(output_image_uint8, cv2.COLOR_GRAY2BGR)
    elif len(output_image_uint8.shape) == 3 and output_image_uint8.shape[2] == 4: # BGRA input (uint8 ndarray)
        output_bgr = cv2.cvtColor(output_image_uint8, cv2.COLOR_BGRA2BGR)
    elif len(output_image_uint8.shape) == 3 and output_image_uint8.shape[2] == 3: # BGR input (uint8 ndarray)
        output_bgr = output_image_uint8 # Keep as is
    else:
        # Fallback if input image format is unexpected after ensure_uint8_image
        print(f"Warning: Cannot convert image with shape {output_image_uint8.shape} to BGR for drawing circles. Returning original image copy.")
        # Convert image to ndarray for copy if it's not already an ndarray type with .copy()
        return np.asarray(image).copy()

    # Check if circles were found and process them if they are a valid MatLike (expected ndarray)
    # HoughCircles returns None if no circles are found, or an array (typically float32).
    # Use 'is not None' check which is necessary runtime logic.
    # Check is needed runtime, ignore diagnostic.
    if circles_mat_result is not None: # type: ignore[reportUnnecessaryComparison]
        # Convert the result to a numpy array for easier processing and shape checks
        # This handles potential cv2.Mat/UMat results from HoughCircles.
        # Use np.asarray to get an ndarray view/copy from any MatLike
        circles_ndarray: np.ndarray[Any, Any] = np.asarray(circles_mat_result)

        # Check if the array has the expected shape (1, N, 3) and floating point dtype
        # Use np.issubdtype for robust dtype check against floating categories
        if circles_ndarray.ndim == 3 and circles_ndarray.shape[0] == 1 and circles_ndarray.shape[2] == 3 and np.issubdtype(circles_ndarray.dtype, np.floating):
            # The data is [center_x, center_y, radius] and is typically float32.
            # Cast to NDArrayFloating for better type handling
            circles_float_array: NDArrayFloating = cast(NDArrayFloating, circles_ndarray)

            # Round to nearest integer and convert to int32 for pixel coordinates/radius
            # OpenCV drawing functions typically accept int coordinates. Using int32 is standard.
            # The result of np.around is float, astype(np.uint8) handles saturation
            # Diagnostic reportUnknownMemberType on around() call likely due to return type ambiguity. Ignore.
            circles_processed: NDArrayInt32 = np.around(circles_float_array).astype(np.int32) # type: ignore[reportUnknownMemberType]

            # Iterate through the detected circles (rows in the 2nd dimension)
            # The shape is (1, N, 3), so circles are in the second dimension (index 1)
            # Type hint for the loop variable is inferred correctly. No comment needed.
            for circle_params_row in circles_processed[0, :]:
                 # Extracting elements from the row and casting to int for clarity and pixel coordinates
                 # Accessing elements from the array slice (np.ndarray[Any, dtype[np.int32]])
                 # Ensure indices are within bounds (already checked by shape == (1, N, 3))
                 center_x = int(circle_params_row[0])
                 center_y = int(circle_params_row[1])
                 radius_val = int(circle_params_row[2])

                 # Ensure coordinates and radius are valid before drawing
                 # Check if coordinates are within image bounds and radius is positive
                 # Access shape safely on the ndarray representation of output_bgr
                 output_bgr_ndarray = np.asarray(output_bgr) # Convert to ndarray for shape access
                 if radius_val > 0 and center_y >= 0 and center_x >= 0 and \
                    center_y < output_bgr_ndarray.shape[0] and center_x < output_bgr_ndarray.shape[1]:
                    try:
                        # cv2.circle accepts MatLike, tuple[int,int] center, int radius, tuple[int,int,int] color, int thickness
                        cv2.circle(output_bgr, (center_x, center_y), 1, (0, 100, 100), 2) # Teal
                        # Draw circle outline
                        cv2.circle(output_bgr, (center_x, center_y), radius_val, (255, 0, 255), 2) # Purple
                    except cv2.error as draw_err:
                        print(f"Error drawing circle at ({center_x},{center_y}) with radius {radius_val}: {draw_err}")
                        pass # Continue drawing other circles

        elif np.asarray(circles_mat_result).size > 0: # Handle cases where circles_mat_result is an array but not shape (1, N, 3)
            print(f"Warning: HoughCircles returned array with unexpected shape: {np.asarray(circles_mat_result).shape}. Expected (1, N, 3) with float dtype.")
        # The unnecessary isinstance check has been removed as np.asarray handles it.

    return output_bgr # Return the image with circles drawn

def apply_find_contours(image: CvMatLike, params: Dict[str, ParamValue]) -> CvMatLike:
    """Finds contours in a binary image and draws them."""
    # Find Contours typically works best on binary images (uint8, 0 or 255).
    # Use _ensure_grayscale_uint8, then check/force binary if needed.
    gray_input_to_process = _ensure_grayscale_uint8(image)
    # Check is needed runtime, ignore diagnostic.
    if gray_input_to_process is None: # type: ignore[reportUnnecessaryComparison]
        print("Find Contours skipped: input not convertible to grayscale uint8, returning original image copy.")
        # Convert image to ndarray for copy if it's not already an ndarray type with .copy()
        return np.asarray(image).copy()

    # Ensure the image is actually binary (only 0 and 255) before finding contours
    # np.unique works on ndarray. gray_input_to_process is NDArrayUInt8.
    unique_vals = np.unique(gray_input_to_process)
    # Check if there are more than 2 unique values or if values are not 0 and 255
    is_not_binary = len(unique_vals) > 2 or not np.all(np.isin(unique_vals, [0, 255]))

    binary_image_for_contours = gray_input_to_process # Start with the grayscale uint8 image
    if is_not_binary:
        # If not binary, apply Otsu's thresholding as a preprocessing step
        print("Warning: Input to Find Contours is not binary. Applying Otsu's threshold.")
        try:
            # cv2.threshold accepts MatLike, returns tuple[float, MatLike]
            # The output MatLike (binary_image_thresh) will be uint8.
            _, binary_image_thresh = cv2.threshold(gray_input_to_process, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Update the image used for contour finding.
            binary_image_for_contours = binary_image_thresh
        except Exception as thresh_e:
            print(f"Error applying Otsu threshold: {thresh_e}. Skipping contour finding.")
            # Convert image to ndarray for copy if it's not already an ndarray type with .copy()
            return np.asarray(image).copy() # Return original if thresholding fails

    # Get parameters for contour finding mode and method
    # Ensure default values are strings matching the keys
    mode_str = str(params.get("mode", "external")).lower()
    method_str = str(params.get("method", "simple")).lower()

    # Define map for string names to OpenCV enums
    cv_mode_map: Dict[str, int] = {"external": cv2.RETR_EXTERNAL, "list": cv2.RETR_LIST, "ccomp": cv2.RETR_CCOMP, "tree": cv2.RETR_TREE}
    cv_method_map: Dict[str, int] = {"simple": cv2.CHAIN_APPROX_SIMPLE, "none": cv2.CHAIN_APPROX_NONE}

    # Get the OpenCV enum values, defaulting if string is not recognized
    cv_mode: int = cv_mode_map.get(mode_str, cv2.RETR_EXTERNAL)
    cv_method: int = cv_method_map.get(method_str, cv2.CHAIN_APPROX_SIMPLE)

    contours: List[CvMatLike] = [] # Initialize empty list for contours
    # Find contours. findContours returns tuple[List[MatLike], MatLike] (or similar)
    # We expect the first element to be a list of contours (CvMatLike objects, typically ndarrays)
    try:
        # findContours expects binary, single-channel, 8-bit input. binary_image_for_contours is NDArrayUInt8 (MatLike compatible).
        # It returns a tuple, typically (contours, hierarchy). We only need contours.
        # Diagnostic reportUnknownMemberType on findContours stub likely due to generic MatLike in return tuple. Ignore.
        contours_tuple = cv2.findContours(binary_image_for_contours, cv_mode, cv_method) # type: ignore[reportUnknownMemberType]

        # The first element of the tuple is the list of contours.
        # Contours are typically lists of ndarrays (int32 or float32 depending on method).
        # Cast the result to a List of np.ndarray[Any, Any] which is more precise and subtype of List[CvMatLike]
        contours = cast(List[np.ndarray[Any, Any]], contours_tuple[0])

    except cv2.error as cv_err:
        print(f"OpenCV error finding contours: {cv_err}. Returning original image copy."); return np.asarray(image).copy()
    except Exception as find_e:
        print(f"Generic error finding contours: {find_e}. Returning original image copy."); return np.asarray(image).copy()

    # Create a color output image for drawing contours, converting the *original* input image.
    # Ensure original image is uint8 and contiguous
    output_viz_image_uint8: NDArrayUInt8 = _ensure_uint8_image(image.copy())

    output_viz_bgr: CvMatLike # Declare type for the color image

    # Convert the (now uint8) input image to BGR if necessary
    # Check shape on the ndarray result of _ensure_uint8_image
    if len(output_viz_image_uint8.shape) == 2: # Grayscale input
        output_viz_bgr = cv2.cvtColor(output_viz_image_uint8, cv2.COLOR_GRAY2BGR)
    elif len(output_viz_image_uint8.shape) == 3 and output_viz_image_uint8.shape[2] == 1: # Grayscale (3D, 1 channel)
         output_viz_bgr = cv2.cvtColor(output_viz_image_uint8, cv2.COLOR_GRAY2BGR)
    elif len(output_viz_image_uint8.shape) == 3 and output_viz_image_uint8.shape[2] == 4: # BGRA input
        output_viz_bgr = cv2.cvtColor(output_viz_image_uint8, cv2.COLOR_BGRA2BGR)
    elif len(output_viz_image_uint8.shape) == 3 and output_viz_image_uint8.shape[2] == 3: # BGR input
        output_viz_bgr = output_viz_image_uint8 # Keep as is
    else:
        # Fallback: Convert the pre-processed binary image to BGR for visualization
        print(f"Warning: Cannot convert input image shape {output_viz_image_uint8.shape} to BGR for drawing. Drawing on binary->BGR conversion.")
        # Convert the binary_image_for_contours (NDArrayUInt8) to BGR
        output_viz_bgr = cv2.cvtColor(binary_image_for_contours, cv2.COLOR_GRAY2BGR)

    # Get parameters for area filtering
    min_area_filter = to_float_robust(params.get("min_area_filter", 0), 0.0)
    # Check if max_area_filter is 0, treat as infinity
    max_area_param = params.get("max_area_filter", 0.0) # Ensure default is float to match type
    # Convert to float robustly, then check against 0.0
    max_area_filter = to_float_robust(max_area_param, 0.0)
    if max_area_filter == 0.0:
        max_area_filter = float('inf')

    filtered_contours: List[CvMatLike] = []

    # Apply area filtering if filtering is enabled (min_area > 0 or max_area is not infinity)
    # Use epsilon for comparing floats if needed, but direct check against 0 and inf is ok here.
    if min_area_filter > 0.0 or max_area_filter < float('inf'):
         # Iterate through the list of np.ndarray contours
         for cnt in contours: # cnt is np.ndarray[Any, Any] (subtype of CvMatLike)
            try:
                # cv2.contourArea accepts MatLike (contour), returns float
                area: float = cv2.contourArea(cnt)
                if min_area_filter <= area <= max_area_filter:
                    # Append the original contour object
                    filtered_contours.append(cnt)
            except Exception as e:
                 print(f"Error calculating contour area: {e}")
                 pass # Skip this contour if area calculation fails
    else:
        # No filtering, keep all contours. Ensure it's a List.
        filtered_contours = list(contours)

    # Draw the filtered contours on the color image
    # cv2.drawContours accepts MatLike output image, List of MatLike contours, etc.
    try:
        # output_viz_bgr is CvMatLike, filtered_contours is List[CvMatLike]
        # drawContours returns the modified image (MatLike, same as output_viz_bgr).
        return cv2.drawContours(output_viz_bgr, filtered_contours, -1, (0, 255, 0), 1) # Green contours
    except cv2.error as cv_err:
        print(f"OpenCV error drawing contours: {cv_err}"); return np.asarray(image).copy()
    except Exception as e:
        print(f"Generic error drawing contours: {e}"); return np.asarray(image).copy()


def apply_filter_contours(image: CvMatLike, params: Dict[str, ParamValue]) -> CvMatLike:
    """Filters contours based on various geometric properties and draws them."""
    # Filter Contours typically works best on binary images.
    gray_input_to_process = _ensure_grayscale_uint8(image)
    # Check is needed runtime, ignore diagnostic.
    if gray_input_to_process is None: # type: ignore[reportUnnecessaryComparison]
        print("Filter Contours skipped: input not convertible to grayscale uint8, returning original image copy.")
        # Convert image to ndarray for copy if it's not already an ndarray type with .copy()
        return np.asarray(image).copy()

    # Ensure binary before finding contours
    unique_vals = np.unique(gray_input_to_process)
    is_not_binary = len(unique_vals) > 2 or not np.all(np.isin(unique_vals, [0, 255]))

    binary_image_for_contours = gray_input_to_process
    if is_not_binary:
        print("Warning: Input to Filter Contours is not binary. Applying Otsu's threshold.")
        try:
            _, binary_image_thresh = cv2.threshold(binary_image_for_contours, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_image_for_contours = binary_image_thresh
        except Exception as thresh_e:
             print(f"Error applying Otsu threshold: {thresh_e}. Skipping contour finding.")
             # Convert image to ndarray for copy if it's not already an ndarray type with .copy()
             return np.asarray(image).copy()

    contours_filter: List[CvMatLike] = [] # Initialize empty list
    # Find contours in the binary image. RETR_EXTERNAL is common for object filtering.
    try:
        # findContours returns tuple[List[MatLike], MatLike]
        # Cast to List[np.ndarray[Any, Any]] as findContours typically returns ndarrays
        # Diagnostic reportUnknownMemberType on findContours stub likely due to generic MatLike in return tuple. Ignore.
        contours_filter_tuple = cv2.findContours(binary_image_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # type: ignore[reportUnknownMemberType]
        # The first element is the list of contours. Cast it.
        contours_filter = cast(List[np.ndarray[Any, Any]], contours_filter_tuple[0])
    except cv2.error as cv_err:
        print(f"OpenCV error finding contours for filtering: {cv_err}. Returning original image copy."); return np.asarray(image).copy()
    except Exception as find_e:
        print(f"Generic error finding contours for filtering: {find_e}. Returning original image copy."); return np.asarray(image).copy()

    # Create a color output image. Drawing on the binary image converted to BGR.
    # Convert binary_image_for_contours (NDArrayUInt8) to BGR
    # cv2.cvtColor returns MatLike (uint8 ndarray)
    output_viz_image_bgr: CvMatLike = cv2.cvtColor(binary_image_for_contours, cv2.COLOR_GRAY2BGR)

    # Get filtering parameters
    min_area = to_float_robust(params.get("min_area", 100), 100.0)
    max_area = to_float_robust(params.get("max_area", 100000), 100000.0)
    min_aspect_ratio = to_float_robust(params.get("min_aspect_ratio", 0.2), 0.2)
    max_aspect_ratio = to_float_robust(params.get("max_aspect_ratio", 5.0), 5.0)
    min_circularity = to_float_robust(params.get("min_circularity", 0.3), 0.3)
    max_circularity = to_float_robust(params.get("max_circularity", 1.2), 1.2)

    final_contours_to_draw: List[CvMatLike] = []
    # Iterate and filter contours
    for contour in contours_filter: # contour is np.ndarray[Any, Any] (subtype of CvMatLike)
        try:
            # cv2.contourArea accepts MatLike (contour), returns float
            area: float = cv2.contourArea(contour)
            if not (min_area <= area <= max_area): continue

            # Get bounding rectangle and calculate aspect ratio
            # cv2.boundingRect accepts MatLike, returns tuple[int, int, int, int]
            _x_coord, _y_coord, w_val, h_val = cv2.boundingRect(contour)
            if w_val == 0 or h_val == 0:
                #print(f"Warning: Zero dimension bounding box ({w_val}x{h_val}) for a contour. Skipping aspect ratio filter.") # Too noisy
                continue # Avoid division by zero
            aspect_ratio = float(w_val) / h_val
            if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio): continue

            # Calculate circularity
            # cv2.arcLength accepts MatLike, returns float
            perimeter: float = cv2.arcLength(contour, True)
            if perimeter == 0:
                # Circularity is undefined or infinite for zero perimeter, skip
                #print("Warning: Zero perimeter contour. Skipping circularity filter.") # Too noisy
                continue
            # Calculate circularity, ensuring non-zero perimeter
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
            if not (min_circularity <= circularity <= max_circularity): continue

            # If all filters pass, add the contour to the list for drawing
            final_contours_to_draw.append(contour)

        except Exception as e:
            print(f"Error filtering contour: {e}. Skipping this contour.")
            pass # Continue filtering other contours

    # Draw the filtered contours
    # cv2.drawContours accepts MatLike output image, List of MatLike contours, etc.
    try:
        # output_viz_image_bgr is CvMatLike, final_contours_to_draw is List[CvMatLike]
        # drawContours returns the modified image (MatLike, same as output_viz_image_bgr).
        return cv2.drawContours(output_viz_image_bgr, final_contours_to_draw, -1, (0, 0, 255), 2) # Red contours
    except cv2.error as cv_err:
        print(f"OpenCV error drawing filtered contours: {cv_err}"); return np.asarray(image).copy()
    except Exception as e:
        print(f"Generic error drawing filtered contours: {e}"); return np.asarray(image).copy()


def apply_morphological_op(image: CvMatLike, params: Dict[str, ParamValue]) -> CvMatLike:
    """Applies morphological operations (erode, dilate, open, close)."""
    # Ensure default values are strings matching the keys
    op_type_str = str(params.get("op_type", "erode")).lower()
    kernel_shape_str = str(params.get("kernel_shape", "rect")).lower()
    ksize_w = to_int_robust(params.get("ksize_w", 5), 5)
    ksize_h = to_int_robust(params.get("ksize_h", 5), 5)
    iterations = to_int_robust(params.get("iterations", 1), 1)

    # Kernel size must be positive
    if ksize_w <= 0: ksize_w = 1
    if ksize_h <= 0: ksize_h = 1

    # Define map for string names to OpenCV enums
    op_type_map: Dict[str, int] = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE, "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}
    kernel_shape_map: Dict[str, int] = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}

    cv_op_type: int = op_type_map.get(op_type_str, cv2.MORPH_ERODE)
    cv_kernel_shape: int = kernel_shape_map.get(kernel_shape_str, cv2.MORPH_RECT)

    # Get structuring element (kernel)
    # getStructuringElement accepts tuple[int, int] for size, returns MatLike (uint8 ndarray)
    # It returns a Mat (which is MatLike), but effectively a uint8 ndarray kernel.
    kernel: CvMatLike = cv2.getStructuringElement(cv_kernel_shape, (ksize_w, ksize_h))

    # Morphology operations often expect binary or grayscale uint8.
    # Ensure the input is uint8. Color images are supported but treated channel-wise.
    # _ensure_uint8_image returns NDArrayUInt8 (CvMatLike subtype)
    img_to_morph: CvMatLike = _ensure_uint8_image(image)

    # cv2.morphologyEx accepts MatLike and returns MatLike (same type as input img_to_morph)
    # img_to_morph is CvMatLike, kernel is CvMatLike. Returns CvMatLike.
    try:
        return cv2.morphologyEx(img_to_morph, cv_op_type, kernel, iterations=iterations)
    except cv2.error as cv_err:
        print(f"OpenCV error in Morphological Op: {cv_err}"); return np.asarray(image).copy()
    except Exception as e:
        print(f"Generic error in Morphological Op: {e}"); return np.asarray(image).copy()


def apply_undistort(image: CvMatLike, params: Dict[str, ParamValue]) -> CvMatLike:
    """Corrects lens distortion using provided camera matrix and distortion coefficients."""
    k_matrix_str = str(params.get("camera_matrix_str", "1000,0,320;0,1000,240;0,0,1"))
    d_coeffs_str = str(params.get("dist_coeffs_str", "0,0,0,0,0"))
    alpha = to_float_robust(params.get("alpha", 0.0), 0.0)

    camera_matrix: Optional[NDArrayFloat32] = None
    dist_coeffs_arr: NDArrayFloat32 = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32) # Initialize with default float32 array

    # Parse camera matrix string
    try:
        k_str_rows = k_matrix_str.strip().split(';')
        # Convert rows/cols to float, handle potential errors
        k_list_of_lists: List[List[float]] = []
        for row_str in k_str_rows:
             try:
                 k_list_of_lists.append([to_float_robust(x.strip()) for x in row_str.split(',')])
             except Exception:
                 print(f"Warning: Skipping invalid row in camera matrix string: '{row_str}'")
                 continue # Skip this row

        # Create a temporary array from the list of lists
        camera_matrix_temp = np.array(k_list_of_lists, dtype=np.float32)
        # Validate shape (3x3)
        if camera_matrix_temp.shape == (3,3):
            camera_matrix = camera_matrix_temp
        else:
            print(f"Warning: Invalid camera matrix shape {camera_matrix_temp.shape}. Expected (3,3).")
            camera_matrix = None # Ensure it's None if shape is wrong
    except Exception as e:
        print(f"Error parsing camera matrix string: {e}")
        camera_matrix = None # camera_matrix remains None on parsing error

    # Parse distortion coefficients string
    try:
        d_coeffs_str_clean = d_coeffs_str.strip()
        if not d_coeffs_str_clean or d_coeffs_str_clean.lower() == "none":
            # Default to 5 zero coefficients if 'none' or empty
            dist_coeffs_arr = np.array([0.0,0.0,0.0,0.0,0.0], dtype=np.float32)
        else:
            # Convert comma-separated values to float list
            d_list = [to_float_robust(x.strip()) for x in d_coeffs_str_clean.split(',')]
            dist_coeffs_arr_temp = np.array(d_list, dtype=np.float32)
            # OpenCV accepts 4, 5, 8, 12, 14 distortion coefficients
            valid_d_lengths = [4, 5, 8, 12, 14]
            if dist_coeffs_arr_temp.size in valid_d_lengths:
                 dist_coeffs_arr = dist_coeffs_arr_temp
            else:
                 print(f"Warning: Invalid distortion coefficients count {dist_coeffs_arr_temp.size}. Expected one of {valid_d_lengths}. Using [0,0,0,0,0].")
                 dist_coeffs_arr = np.array([0.0,0.0,0.0,0.0,0.0], dtype=np.float32) # Fallback
    except Exception as e:
        print(f"Error parsing distortion coefficients string: {e}")
        dist_coeffs_arr = np.array([0.0,0.0,0.0,0.0,0.0], dtype=np.float32) # Fallback to zero coeffs on error

    # Convert input image to ndarray for robust shape and size access
    img_ndarray: np.ndarray[Any, Any] = np.asarray(image)

    # Check if essential parameters are available and image is not empty/degenerate
    # camera_matrix is Optional, dist_coeffs_arr is guaranteed not None due to initialization/fallback.
    # Check is needed runtime, ignore diagnostic.
    if camera_matrix is None or img_ndarray.size == 0: # type: ignore[reportUnnecessaryComparison]
        print("Warning: Undistort skipped due to invalid camera parameters or empty image.")
        # Convert image to ndarray for copy if it's not already an ndarray type with .copy()
        return np.asarray(image).copy()

    frame_height, frame_width = img_ndarray.shape[:2]
    if frame_width == 0 or frame_height == 0:
        print("Warning: Undistort skipped due to zero dimension image.")
        # Convert image to ndarray for copy if it's not already an ndarray type with .copy()
        return np.asarray(image).copy()

    try:
        # OpenCV functions getOptimalNewCameraMatrix and initUndistortRectifyMap often prefer float64
        # Cast the float32 numpy arrays to float64 for these specific calls
        # Cast is safe as they are np.ndarray subtypes
        # camera_matrix is NDArrayFloat32 (if not None), dist_coeffs_arr is NDArrayFloat32
        camera_matrix_cv: NDArrayFloating = cast(NDArrayFloating, camera_matrix.astype(np.float64))
        dist_coeffs_cv: NDArrayFloating = cast(NDArrayFloating, dist_coeffs_arr.astype(np.float64))

        # getOptimalNewCameraMatrix returns: optimalCameraMatrix (MatLike, float64), roi (Rect tuple[int,int,int,int])
        # Provide image size (width, height) and desired new image size (same as original for full result)
        # Diagnostic reportUnknownMemberType - Pyright stub for Rect is Sequence[int]. Cast the result tuple explicitly. Ignore method call stub.
        optimal_camera_matrix_cv_raw: CvMatLike # The MatLike result
        roi_rect_raw: Sequence[int] # The Rect result, often typed as Sequence[int] in stubs
        optimal_camera_matrix_cv_raw, roi_rect_raw = cast( # type: ignore[reportUnknownMemberType]
            tuple[CvMatLike, Sequence[int]], # Explicitly cast the tuple returned by cv2.getOptimalNewCameraMatrix
            cv2.getOptimalNewCameraMatrix(
                camera_matrix_cv, dist_coeffs_cv, (frame_width, frame_height), alpha, (frame_width, frame_height)
            )
        )
        # Now cast the raw Sequence[int] to the expected tuple[int, int, int, int] after checking length/validity if needed,
        # or rely on cv2's output being a fixed-length Rect. The cast below asserts the expected structure.
        # Add a check to be safer, although OpenCV typically returns a valid Rect.
        if len(roi_rect_raw) == 4:
            roi_rect: tuple[int, int, int, int] = cast(tuple[int, int, int, int], roi_rect_raw)
        else:
             print(f"Warning: getOptimalNewCameraMatrix returned unexpected ROI format: {roi_rect_raw}. Skipping cropping.")
             # Set ROI to full image if extraction fails
             roi_rect = (0, 0, frame_width, frame_height)

        optimal_camera_matrix_cv: CvMatLike = optimal_camera_matrix_cv_raw # Assign the MatLike part

        # initUndistortRectifyMap requires camera matrix in float32 or float64.
        # R is typically identity (3x3) for basic undistortion. Use float64 to match others.
        r_matrix: NDArrayFloating = np.eye(3, dtype=np.float64) # Identity matrix as float64

        # Map x and y pixel coordinates from distorted to undistorted.
        # mapx, mapy will be MatLike (float32 ndarrays)
        mapx: CvMatLike
        mapy: CvMatLike
        # Diagnostic reportUnknownMemberType on initUndistortRectifyMap stub likely due to generic MatLike return type. Ignore.
        mapx, mapy = cv2.initUndistortRectifyMap( # type: ignore[reportUnknownMemberType]
            camera_matrix_cv,       # Camera matrix K (float64 MatLike)
            dist_coeffs_cv,         # Distortion coefficients D (float64 MatLike)
            r_matrix,               # Rectification transform R (float64 MatLike)
            optimal_camera_matrix_cv, # New camera matrix (float64 MatLike from prev step)
            (frame_width, frame_height), # Size of the undistorted image (width, height)
            cv2.CV_32FC1            # Type of mapx, mapy (float32, single channel)
        )

        # remap applies the transformation maps to the input image.
        # It accepts CvMatLike (the original image) and returns CvMatLike (same dtype as input).
        # Pass the original CvMatLike image directly.
        # image is CvMatLike, mapx/mapy are CvMatLike. Returns CvMatLike.
        # Diagnostic reportUnknownMemberType on remap stub likely due to generic MatLike return type. Ignore.
        undistorted_image: CvMatLike = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR) # type: ignore[reportUnknownMemberType]

        # Crop the image based on the ROI returned by getOptimalNewCameraMatrix
        x_r, y_r, w_r, h_r = roi_rect # Unpacking the casted tuple

        # Convert undistorted image to ndarray for robust shape and slicing checks
        undistorted_img_ndarray: np.ndarray[Any, Any] = np.asarray(undistorted_image)

        # Check if the ROI is valid and within the bounds of the undistorted image
        # Access shape attributes safely after converting to ndarray
        if w_r > 0 and h_r > 0 and \
           0 <= y_r < undistorted_img_ndarray.shape[0] and 0 <= x_r < undistorted_img_ndarray.shape[1] and \
           y_r + h_r <= undistorted_img_ndarray.shape[0] and x_r + w_r <= undistorted_img_ndarray.shape[1]:

            # Ensure slicing indices are integers (already int from ROI tuple cast, but explicit for clarity)
            y_start, y_end = int(y_r), int(y_r + h_r)
            x_start, x_end = int(x_r), int(x_r + w_r)

            # Return a copy of the sliced portion to ensure subsequent operations don't affect this result's underlying data
            # Slicing an ndarray results in an ndarray view, which is CvMatLike. Copying is needed.
            return undistorted_img_ndarray[y_start:y_end, x_start:x_end].copy()

        else:
             # If ROI is degenerate or out of bounds, return the full undistorted image
             if w_r <= 0 or h_r <= 0:
                 print(f"Warning: ROI width or height is zero or negative {roi_rect}. Returning full undistorted image.")
             else: # ROI coordinates are out of bounds
                 print(f"Warning: ROI {roi_rect} is out of bounds for undistorted image shape {undistorted_img_ndarray.shape}. Returning full undistorted image.")
             # Return a copy of the full image
             return undistorted_image.copy()

    except cv2.error as cv_err:
        print(f"OpenCV error during undistortion: {cv_err}"); return np.asarray(image).copy()
    except Exception as e:
        print(f"Generic error during undistortion: {e}"); return np.asarray(image).copy()


def apply_resize(image: CvMatLike, params: Dict[str, ParamValue]) -> CvMatLike:
    """Resizes the image to specified dimensions."""
    target_width = to_int_robust(params.get("width", 640), 640)
    target_height = to_int_robust(params.get("height", 480), 480)

    # Map interpolation string to OpenCV enum
    interpolation_methods: Dict[str, int] = { # Explicitly type the dict
        "nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC, "lanczos4": cv2.INTER_LANCZOS4
    }
    interpolation_str = str(params.get("interpolation", "linear")).lower()
    interpolation: int = interpolation_methods.get(interpolation_str, cv2.INTER_LINEAR)

    # Convert image to ndarray-like for shape access robustness
    img_ndarray_check: np.ndarray[Any, Any] = np.asarray(image)

    # Check if resize is necessary (current size matches target) and dimensions are valid
    # Access shape attributes on the ndarray
    # Defensive check for non-empty shape tuple
    if len(img_ndarray_check.shape) >= 2 and img_ndarray_check.shape[1] == target_width and img_ndarray_check.shape[0] == target_height:
        # No change needed, return a copy of the original input image (or the ndarray representation)
        return np.asarray(image).copy()

    if target_width <= 0 or target_height <= 0:
        print(f"Warning: Invalid resize dimensions ({target_width}, {target_height}). Returning original image copy.")
        # Convert image to ndarray for copy if it's not already an ndarray type with .copy()
        return np.asarray(image).copy()

    # cv2.resize accepts MatLike and returns MatLike (typically ndarray with same dtype as input)
    # Pass the original CvMatLike image directly. image is CvMatLike. Returns CvMatLike.
    try:
        return cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    except cv2.error as cv_err:
        print(f"OpenCV error in Resize: {cv_err}"); return np.asarray(image).copy()
    except Exception as e:
        print(f"Generic error in Resize: {e}"); return np.asarray(image).copy()


# --- ImageOperation Class ---
# Represents a single image processing step in the pipeline.
class ImageOperation:
    # Added type hints for instance attributes
    id: uuid.UUID
    base_name: str
    op_type_name: str
    function_ref: Optional[ProcessingFunction]
    params: Dict[str, ParamConfigValue]
    is_enabled: bool

    def __init__(self, base_name: str,
                 function_ref: Optional[ProcessingFunction] = None,
                 default_params: Optional[Dict[str, ParamConfigValue]] = None,
                 is_enabled: bool = True,
                 op_type_name: Optional[str] = None):
        # Validate inputs - reportUnnecessaryIsInstance diagnostics removed as types are in signature
        # The check `isinstance(function_ref, Callable)` is redundant if function_ref is Optional[ProcessingFunction]
        # as ProcessingFunction is Callable. Removing the redundant check.
        # if function_ref is not None and not callable(function_ref): raise TypeError("function_ref must be a callable or None")


        self.id = uuid.uuid4() # Unique ID for this operation instance
        self.base_name = base_name # User-facing name (can be edited)
        # Internal type name, used to find the processing function and default config
        # Use base_name as fallback if op_type_name is None or empty string
        self.op_type_name = op_type_name if op_type_name else base_name # Use 'else base_name' for empty string case
        # Reference to the actual processing function
        self.function_ref = function_ref
        # Parameters for this operation instance. Copy default_params if provided.
        self.params = {}
        # Check is needed runtime, ignore diagnostic.
        if default_params is not None: # type: ignore[reportUnnecessaryComparison]
             # Iterate and copy each ParamConfigValue dictionary
             for k, v_config in default_params.items():
                  # v_config is ParamConfigValue, which is a TypedDict (dictionary subclass)
                  self.params[k] = v_config.copy() # Shallow copy of the inner dictionary


        self.is_enabled = is_enabled # Whether this operation is currently active

    @override
    def __str__(self) -> str:
        """String representation, primarily for display."""
        return self.base_name

    def get_params_for_processing(self) -> Dict[str, ParamValue]:
        """Extracts just the parameter values suitable for passing to the processing function."""
        # Use a dictionary comprehension to get name -> value pairs
        # Ensure 'value' key exists, although ParamConfigValue TypedDict implies it will.
        # This dictionary comprehension is safe because self.params is typed Dict[str, ParamConfigValue]
        # and ParamConfigValue is a TypedDict with 'value' as a required key.
        return {name: props["value"] for name, props in self.params.items()}

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the operation state to a dictionary."""
        # Note: UUID is directly serializable by json.dump when default=str is used,
        # but explicitly converting to string is clearer.
        return {
            "id": str(self.id), # Convert UUID to string for serialization
            "base_name": self.base_name,
            "op_type_name": self.op_type_name,
            # TypedDicts work like dicts and are JSON serializable if their values are.
            # ParamConfigValue contains str, int, float, bool, Dict[str, ParamValue], which are serializable.
            "params": self.params,
            "is_enabled": self.is_enabled
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], processing_functions: Dict[str, ProcessingFunction]) -> 'ImageOperation':
        """Creates an ImageOperation instance from a dictionary (e.g., loaded from JSON)."""
        # data is expected to be Dict[str, Any] from the caller

        # Get and validate base_name (required)
        base_name_raw: Any = data.get("base_name")
        # Necessary because source is Any. Keep check, ignore diagnostic.
        if not isinstance(base_name_raw, str): # type: ignore[reportUnnecessaryIsInstance]
             print(f"Error loading operation: 'base_name' missing or not string. Data: {data}. Using 'Unknown Operation'.")
             base_name: str = "Unknown Operation"
        else:
            base_name = base_name_raw # base_name is now correctly str

        # Get and validate op_type_name (falls back to base_name if missing)
        op_type_name_raw: Any = data.get("op_type_name")
        final_op_type_name: str = base_name # Default to base_name
        # Necessary because source is Any. Keep check, ignore diagnostic.
        if isinstance(op_type_name_raw, str): # type: ignore[reportUnnecessaryIsInstance]
            final_op_type_name = op_type_name_raw
        elif op_type_name_raw is not None:
             print(f"Warning: Loaded operation '{base_name}' has invalid 'op_type_name' (not a string): {op_type_name_raw}.")

        # Look up the processing function by op_type_name
        func_ref: Optional[ProcessingFunction] = processing_functions.get(final_op_type_name)
        # Check is needed runtime, ignore diagnostic.
        if func_ref is None: # type: ignore[reportUnnecessaryComparison]
            print(f"Warning: No processing function registered for loaded op_type_name '{final_op_type_name}' (base name '{base_name}'). This operation will be inactive.")

        # Get and validate params dictionary
        loaded_params_raw: Any = data.get("params", {})
        loaded_params_typed: Dict[str, ParamConfigValue] = {}
        # Necessary because loaded_params_raw is Any. Keep check, ignore diagnostic.
        if isinstance(loaded_params_raw, dict): # type: ignore[reportUnnecessaryIsInstance]
            # Iterate through loaded params, validate structure and cast values
            # Cast loaded_params_raw to Dict[str, Any] before iterating over its items.
            loaded_params_as_dict_str_any = cast(Dict[str, Any], loaded_params_raw)
            for p_name_any, p_props_any in loaded_params_as_dict_str_any.items():
                # Validate parameter name is string and properties is a dictionary
                # Added check for p_name_any as it comes from Any source
                # Necessary because source is Any. Keep check, ignore diagnostic.
                if isinstance(p_name_any, str) and isinstance(p_props_any, dict): # type: ignore[reportUnnecessaryIsInstance]
                    # Basic check for required keys for ParamConfigValue
                    if "type" in p_props_any and "value" in p_props_any:
                        # Cast the parameter properties dictionary to the TypedDict ParamConfigValue
                        # This cast informs Pyright about the expected structure of p_props_any
                        # Note: The type within `p_props_any` (the value of 'value') is still `Any`
                        # and will need robust handling in the UI/processing functions.
                        try:
                             # Add a cast to ParamConfigValue here for better type tracking of the inner dict
                             loaded_params_typed[p_name_any] = cast(ParamConfigValue, p_props_any)
                        except TypeError: # Catch potential issues if data structure isn't exactly as typeddict expects
                             print(f"Warning: Param '{p_name_any}' for op '{base_name}' has incorrect structure during load. Skipping.")
                             pass # Skip this parameter
                    else:
                         print(f"Warning: Param '{p_name_any}' for op '{base_name}' missing 'type' or 'value' field during load. Skipping.")
                else:
                    print(f"Warning: Invalid parameter entry format for op '{base_name}' during load. Entry: {p_name_any}: {p_props_any}. Skipping.")
        # Check is needed runtime, ignore diagnostic.
        elif loaded_params_raw is not None: # type: ignore[reportUnnecessaryComparison]
             print(f"Warning: Loaded operation '{base_name}' has invalid 'params' format (not a dict): {loaded_params_raw}. Using empty params.")


        # Get and validate is_enabled flag
        is_enabled_raw: Any = data.get("is_enabled", True) # Default to True if missing
        # Necessary because source is Any. Keep check, ignore diagnostic.
        is_enabled: bool = bool(is_enabled_raw) # type: ignore[reportUnnecessaryIsInstance] # Robustly convert to bool, bool(Any) is not useless here.

        # Attempt to parse the UUID if present, otherwise generate a new one
        op_id = uuid.uuid4() # Default to new ID
        id_from_data_raw: Any = data.get("id")
        # Necessary because source is Any. Keep check, ignore diagnostic.
        if isinstance(id_from_data_raw, str): # type: ignore[reportUnnecessaryIsInstance]
            try:
                op_id = uuid.UUID(id_from_data_raw)
            except ValueError:
                print(f"Warning: Invalid UUID string '{id_from_data_raw}' for op '{base_name}' during load. Generating new ID.")
                pass # Keep the default new uuid
        # Check is needed runtime, ignore diagnostic.
        elif id_from_data_raw is not None: # type: ignore[reportUnnecessaryComparison]
             print(f"Warning: Loaded operation '{base_name}' has invalid 'id' format (not a string): {id_from_data_raw}. Generating new ID.")


        # Create the ImageOperation instance
        # Pass the validated/typed params
        op = cls(base_name=base_name,
                 op_type_name=final_op_type_name,
                 function_ref=func_ref,
                 default_params=loaded_params_typed,
                 is_enabled=is_enabled)

        # Assign the parsed/generated ID to the instance
        op.id = op_id

        return op

# --- Emitter and ListItem ---
# OperationSignalEmitter is a simple QObject subclass used solely for defining signals
class OperationSignalEmitter(QObject):
    # Signal emitted when an operation's enabled state changes in the list item UI.
    # Emits the operation UUID and the new boolean state.
    operation_enabled_changed = pyqtSignal(uuid.UUID, bool)

# Custom QListWidgetItem to hold an ImageOperation and manage its UI representation.
class OperationListItem(QListWidgetItem):
    # Store the signal connection explicitly for safer disconnection
    _connection_checkbox_state: QMetaObject.Connection

    def __init__(self, operation: ImageOperation, parent_list_widget: QListWidget):
        # Initialize the base QListWidgetItem, associating it with the parent list widget.
        super().__init__(parent_list_widget)

        # Store the ImageOperation instance and parent widget reference.
        self.operation: ImageOperation = operation
        self.parent_list_widget: QListWidget = parent_list_widget # Store reference to the containing list widget

        # Create a signal emitter specific to this item to communicate state changes.
        # The emitter is owned by the list item widget, so it will be cleaned up when the item is removed.
        self.item_widget: QWidget = QWidget()
        # Allow stylesheet from parent list to show through.
        # Setting background to transparent via stylesheet.
        # Diagnostic reportUnknownMemberType on setStyleSheet stub addressed by ignore
        self.item_widget.setStyleSheet("background-color: transparent;") # type: ignore[reportUnknownMemberType]

        # Create a horizontal layout for the widget (checkbox + label).
        self.item_layout: QHBoxLayout = QHBoxLayout(self.item_widget)
        self.item_layout.setContentsMargins(5, 3, 5, 3) # Set margins
        self.item_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter) # Align content

        # Create and configure the checkbox.
        self.checkbox: QCheckBox = QCheckBox()
        # Diagnostic reportUnknownMemberType on setStyleSheet stub addressed by ignore
        self.checkbox.setStyleSheet("QCheckBox { spacing: 5px; background-color: transparent; }") # type: ignore[reportUnknownMemberType]
        self.checkbox.setChecked(self.operation.is_enabled) # Set initial state from operation
        # Connect the stateChanged signal (int) to the handler slot using a lambda.
        # QCheckBox.stateChanged signal emits an integer (0=Unchecked, 2=Checked).
        # Explicitly hint the parameter 'state' in the lambda as int using cast.
        # Diagnostic reportUnknownMemberType on stateChanged.connect stub addressed by ignore
        # Diagnostic reportUnknownLambdaType on lambda parameters addressed by ignore
        self._connection_checkbox_state = self.checkbox.stateChanged.connect( # type: ignore[reportUnknownMemberType]
            lambda state: self.on_checkbox_changed(cast(int, state)) # type: ignore[reportUnknownLambdaType]
        )

        # Create and configure the label.
        self.label: QLabel = QLabel()
        self.update_label_text() # Set initial text (includes index)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred) # Label expands to fill space
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Add checkbox and label to the widget's layout.
        self.item_layout.addWidget(self.checkbox)
        self.item_layout.addWidget(self.label)

        # Set appropriate size hint for the list item based on font metrics.
        # This ensures the row height is correct.
        font_to_use: QFont = self.parent_list_widget.font() if self.parent_list_widget else QApplication.font() # font() stub
        fm = QFontMetrics(font_to_use)
        min_height = fm.height() + 10 # Add some vertical padding
        # Diagnostic reportUnknownMemberType on setMinimumHeight stub addressed by ignore
        self.item_widget.setMinimumHeight(min_height) # type: ignore[reportUnknownMemberType]

        # Use sizeHint() which calculates the preferred size based on layout and contents.
        # Diagnostic reportUnknownMemberType on sizeHint stub addressed by ignore
        self.setSizeHint(self.item_widget.sizeHint()) # type: ignore[reportUnknownMemberType]

        # Assign the custom widget to this list item.
        # Diagnostic reportUnknownMemberType on setItemWidget stub addressed by ignore
        self.parent_list_widget.setItemWidget(self, self.item_widget) # type: ignore[reportUnknownMemberType]

        # Emitter is created and owned by the QWidget, which is owned by the QListWidgetItem.
        # Qt's parent-child memory management should handle deletion.
        self.emitter: OperationSignalEmitter = OperationSignalEmitter(self.item_widget) # Make item_widget the parent


    # Add __del__ method to explicitly disconnect checkbox signal on deletion if needed,
    # though Qt's parent mechanism usually handles this if parentage is set correctly.
    # However, since we store the connection, explicit disconnection is safer.
    def __del__(self) -> None:
        # Ensure the stored connection object is valid before attempting disconnection
        # This check might be redundant if parentage guarantees cleanup, but safer.
        # Use explicit check against QMetaObject.Connection type as it's the stored type
        # Check is needed runtime, ignore diagnostic.
        if isinstance(self._connection_checkbox_state, QMetaObject.Connection): # type: ignore[reportUnnecessaryIsInstance]
             try:
                 # QObject.disconnect(self._connection_checkbox_state) is the correct way
                 # to disconnect a specific connection object.
                 # Diagnostic reportUnknownMemberType on disconnect stub addressed by ignore
                 QObject.disconnect(self._connection_checkbox_state) # type: ignore[reportUnknownMemberType]
             except (TypeError, RuntimeError):
                 # Pass if the connection was already broken or deleted
                 pass
        # If the connection is not a QMetaObject.Connection (unexpected) or was never assigned, do nothing.


    def on_checkbox_changed(self, state_value: int) -> None:
        """Slot handling checkbox state changes, updates operation state and emits signal."""
        # Convert integer state to boolean Checked status.
        is_checked = (Qt.CheckState(state_value) == Qt.CheckState.Checked)
        # Check if the state has actually changed to avoid unnecessary updates/emits
        if self.operation.is_enabled != is_checked:
             # Update the corresponding ImageOperation object's state.
             self.operation.is_enabled = is_checked
             # Emit the signal with the operation's ID and new state.
             # Diagnostic reportUnknownMemberType on emit stub addressed by ignore
             self.emitter.operation_enabled_changed.emit(self.operation.id, is_checked) # type: ignore[reportUnknownMemberType]

    def update_label_text(self) -> None:
        """Updates the label text to include the current row number."""
        # Get the row index of this item in the parent list widget.
        # row() returns -1 if the item is not in the list (shouldn't happen here).
        # Diagnostic reportUnknownMemberType on row stub addressed by ignore
        row = self.parent_list_widget.row(self) # type: ignore[reportUnknownMemberType]
        # Add 1 to row index for 1-based numbering shown to the user.
        prefix = f"{row + 1}. " if row >= 0 else ""
        # Set the label text using the operation's base name.
        # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
        self.label.setText(f"{prefix}{self.operation.base_name}") # type: ignore[reportUnknownMemberType]


# --- Video Capture Thread ---
class VideoThread(QThread):
    # Signal to emit new frames. Emits object because PyQt doesn't natively type NumPy arrays.
    # The receiver slot needs to handle runtime checks and casting.
    # The frame is always NDArrayUInt8 after _ensure_uint8_image in the thread.
    new_frame_signal = pyqtSignal(object) # Emits object which will be NDArrayUInt8
    # Signal for error messages (string).
    error_signal = pyqtSignal(str)
    # Signal emitted when the thread finishes.
    finished_signal = pyqtSignal()

    # Added type hints for instance variables
    video_source: Union[int, str]
    _is_running: bool
    cap: Optional[cv2.VideoCapture]

    def __init__(self, video_source: Union[int, str] = 0, parent: Optional[QObject] = None):
        # Add optional parent argument for Qt object model hierarchy if needed
        super().__init__(parent)
        # Validate video_source type - Diagnostic reportUnnecessaryIsInstance removed.
        # The check `isinstance(video_source, Union[int, str])` is redundant as the type is in the signature.
        # Removing the redundant check.

        self.video_source = video_source # Assign directly if type is correct

        self._is_running = True # Flag to control the run loop
        # Initialize as Optional as capture object creation might fail.
        self.cap = None

    @override
    def run(self) -> None:
        """Main thread execution method for capturing video frames."""
        try:
            # Initialize VideoCapture
            # cv2.VideoCapture can return a non-None object even if it fails to open.
            # The check `isOpened()` is needed.
            # The type hint for the return of VideoCapture is often just VideoCapture, not Optional[VideoCapture].
            # The assignment `self.cap = ...` is fine, but the subsequent `is None` check is necessary runtime.
            # Diagnostic reportUnknownMemberType on VideoCapture stub addressed by ignore
            self.cap = cv2.VideoCapture(self.video_source) # type: ignore[reportUnknownMemberType]

            # Check if the source was opened successfully
            # Runtime can fail and self.cap might be None or an invalid object. Keep the check. Ignore diagnostic.
            # Diagnostic reportUnknownMemberType on isOpened stub addressed by ignore
            if self.cap is None or not self.cap.isOpened(): # type: ignore[reportUnnecessaryComparison, reportUnknownMemberType]
                error_msg = f"Could not open video source: {self.video_source}"
                # Diagnostic reportUnknownMemberType on emit stub addressed by ignore
                self.error_signal.emit(error_msg) # type: ignore[reportUnknownMemberType]
                self._is_running = False # Stop the thread if source fails
                # Return early as capture failed
                return

            # Attempt to get source properties for FPS and frame count
            # cap.get() returns float. Add type hint.
            # Diagnostic reportUnknownMemberType on get stub addressed by ignore
            fps_cap: float = self.cap.get(cv2.CAP_PROP_FPS) # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on get stub addressed by ignore
            total_frames_cap: float = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) # type: ignore[reportUnknownMemberType]

            # Determine if the source is likely a file (check if it's a string path that exists)
            # Ensure self.video_source is treated as str for the check
            is_file_source = isinstance(self.video_source, str) and os.path.isfile(self.video_source)

            # Determine frame delay for playback rate control
            # Default to 30 FPS if FPS is unknown (0) or non-positive
            fps: float = fps_cap if fps_cap > 0.0 else 30.0
            # Calculate delay in milliseconds, ensuring minimum 1ms delay
            frame_delay_ms: int = max(1, int(1000.0 / fps)) # Use 1000.0 for float division

            print(f"Video source opened: {self.video_source}. FPS: {fps_cap:.2f}, Total Frames: {total_frames_cap:.0f}, Frame Delay (ms): {frame_delay_ms}")

            # Main loop for capturing frames
            while self._is_running:
                loop_start_time = time.perf_counter() # Record start time for timing

                # Check if capture object is still valid before read (should be unless unexpected error)
                # Runtime can fail. Keep check and ignore.
                if self.cap is None: # type: ignore[reportUnnecessaryComparison] # Check is needed runtime
                     print("Video capture object became None unexpectedly.")
                     break # Exit loop if cap is unexpectedly None

                # Read a frame from the capture source
                # read() returns tuple[bool, CvMatLike | None]. Expecting CvMatLike if ret is True.
                # Diagnostic reportUnknownMemberType on read stub addressed by ignore
                ret, frame_mat = self.cap.read() # type: ignore[reportUnknownMemberType]

                # Runtime can return None or an empty Mat even if ret is True in some cases. Keep the check. Ignore diagnostic.
                # np.asarray handles MatLike -> ndarray conversion robustly.
                # frame_mat could be None according to stubs. Keep check, ignore diagnostic.
                if ret:
                    if frame_mat is not None and np.asarray(frame_mat).size > 0: # type: ignore[reportUnnecessaryComparison]
                        # Convert the captured frame (CvMatLike) to NDArrayUInt8 consistently
                        # This handles potential varying return types of cv2.read() and ensures uint8 dtype
                        try:
                            # Use the robust conversion helper. _ensure_uint8_image returns NDArrayUInt8.
                            frame_to_emit: NDArrayUInt8 = _ensure_uint8_image(frame_mat)
                            # Emit the processed frame signal (as object for PyQt)
                            # Diagnostic reportUnknownMemberType on emit stub addressed by ignore
                            self.new_frame_signal.emit(frame_to_emit) # type: ignore[reportUnknownMemberType]
                        except Exception as conversion_e:
                            # Handle errors during image conversion before emitting
                            print(f"Error converting captured frame: {conversion_e}")
                            # Diagnostic reportUnknownMemberType on emit stub addressed by ignore
                            self.error_signal.emit(f"Frame conversion error: {conversion_e}") # type: ignore[reportUnknownMemberType]
                            # For now, we continue the loop even on conversion error, but log/report it.
                            pass # Continue loop
                    else:
                        # ret is True, but frame_mat is None or empty. Log this unexpected state.
                         print("Warning: cv2.read returned True but frame is None or empty.")
                         # Diagnostic reportUnknownMemberType on emit stub addressed by ignore
                         self.error_signal.emit("Received None or empty frame from source") # type: ignore[reportUnknownMemberType]


                else: # If ret is False (failed to read frame)
                    # For files or general streams (string sources), assume end of source or persistent error
                    # Also include streams that aren't file paths
                    is_stream_source = isinstance(self.video_source, str) and not os.path.isfile(str(self.video_source)) and not str(self.video_source).isdigit()
                    if is_file_source or is_stream_source:
                        err_msg = "Video file/stream ended or read error."
                        # Diagnostic reportUnknownMemberType on emit stub addressed by ignore
                        self.error_signal.emit(err_msg) # type: ignore[reportUnknownMemberType]
                        self._is_running = False
                        break # Exit loop
                    else: # For cameras (integer source), keep trying, maybe it's temporary frame drop
                        # Wait a bit before retrying camera read
                        # Diagnostic reportUnknownMemberType on msleep stub addressed by ignore
                        self.msleep(200); continue # type: ignore[reportUnknownMemberType]


                # Control frame rate based on calculated delay
                elapsed_time_ms = (time.perf_counter() - loop_start_time) * 1000
                sleep_time_ms = frame_delay_ms - elapsed_time_ms
                if sleep_time_ms > 0:
                    # Use msleep (milliseconds) from QThread
                    # Diagnostic reportUnknownMemberType on msleep stub addressed by ignore
                    self.msleep(int(sleep_time_ms)) # type: ignore[reportUnknownMemberType]


        except Exception as e:
            # Catch any other unexpected exceptions in the thread
            # Diagnostic reportUnknownMemberType on emit stub addressed by ignore
            self.error_signal.emit(f"VideoThread exception: {str(e)}") # type: ignore[reportUnknownMemberType]
        finally:
            # Clean up: Release the video capture object and update state
            # Check is needed runtime, ignore diagnostic.
            if self.cap: # type: ignore[reportUnnecessaryComparison]
                try:
                    # Diagnostic reportUnknownMemberType on release stub addressed by ignore
                    self.cap.release() # Release the resource # type: ignore[reportUnknownMemberType]
                except Exception as release_e: print(f"Error releasing video capture: {release_e}")
            self.cap = None; self._is_running = False # Ensure flag is False on exit
            # Emit finished signal
            # Diagnostic reportUnknownMemberType on emit stub addressed by ignore
            self.finished_signal.emit() # type: ignore[reportUnknownMemberType]

    def stop(self) -> None:
        """Safely signals the video thread to stop."""
        self._is_running = False


# --- Main Application Window Class ---
class ProcessingPipelineApp(QMainWindow):
    # Declare instance variables with type hints for clarity and Pyright
    video_source: Union[int, str] # The current video source identifier
    current_cv_frame: Optional[NDArrayUInt8] # The latest raw frame from the video source (NDArrayUInt8 after conversion)
    original_frame_for_display: Optional[CaptureFrameType] # A copy of the original frame (uint8) for display
    selected_stage_input_frame: Optional[CvMatLike] # Frame before the selected operation
    selected_stage_output_frame: Optional[CvMatLike] # Frame after the selected operation
    operations_data: List[ImageOperation] # List storing the current pipeline operations
    # Parameter panel layout and explanation label are Optional until init_ui is called, though effectively always present after.
    parameter_widgets_area_layout: Optional[QVBoxLayout]
    operation_explanation_label: Optional[QLabel]
    video_thread: Optional[VideoThread] # The background thread for video capture
    first_frame_received: bool # Flag to track if the first frame has arrived
    _connections: List[QMetaObject.Connection] # List to store Qt signal connections for cleanup

    # Static configuration data
    # Corrected the structure of param_explanations within the literal dictionary to match Dict[str, str].
    operation_details: Dict[str, OpExplanationEntry] = { # Corrected dictionary literal structure
        "Undistort": {
                "explanation": "Corrects lens distortion using camera calibration parameters (K, D).\n\n"
                               "<b>How:</b> Uses `cv2.remap` with maps generated from K, D, and alpha. These parameters are typically obtained from camera calibration and configured in the 'Source & Calibration' section or loaded with a pipeline file.\n"
                               "<b>Why:</b> Essential for geometric accuracy if the lens introduces distortion.\n"
                               "<b>When:</b> Typically the very first operation in the pipeline.",
                "param_explanations": {
                    # Note: parameter names here must match the keys in ParamConfigValue within available_op_configs
                    "camera_matrix_str": "<b>Camera Matrix (K):</b> String representation of the 3x3 intrinsic camera matrix. Format: `fx,0,cx;0,fy,cy;0,0,1`. Example: `1000,0,320;0,1000,240;0,0,1`.",
                    "dist_coeffs_str": "<b>Distortion Coefficients (D):</b> String representation of the distortion coefficients vector. Comma-separated values: `k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,tx,ty]]]]`. Number of coefficients can be 4, 5, 8, 12, or 14. Use `0,0,0,0,0` for no distortion.",
                    "alpha": "<b>Alpha (Undistortion):</b> Free scaling parameter (0-1). Controls how much of the undesirable pixels (that would be outside the original image bounds after undistortion) are cropped away.\n  `0`: All black pixels around the valid region are cropped.\n  `1`: All source pixels are retained, even if they are distorted near the edges."
                }
            },
            "Resize": {
                "explanation": "Resizes the image to a specified target width and height using a chosen interpolation method.\n\n"
                               "<b>Why:</b> Used to standardize image dimensions, reduce image size for faster processing, or enlarge images for display.\n"
                               "<b>When to use:</b> Can be applied early to reduce computational load, or later to ensure output matches required dimensions. Note that resizing can introduce artifacts or blur depending on the method and scaling factor.",
                "param_explanations": {
                    "width": "<b>Target Width:</b> The desired width of the output image in pixels.",
                    "height": "<b>Target Height:</b> The desired height of the output image in pixels.",
                    "interpolation": "<b>Interpolation Method:</b> The algorithm used to calculate pixel values in the resized image. Options include `nearest`, `linear` (default), `area` (good for shrinking), `cubic`, and `lanczos4` (good for enlarging)."
                }
            },
            "Grayscale": {
                "explanation": "Converts a color image (BGR or BGRA) to a single-channel grayscale image (typically 8-bit).\n\n"
                               "<b>Why:</b> Simplifies the image data, reducing computational load for subsequent processing steps. Many computer vision algorithms (e.g., edge detection, thresholding, some feature descriptors) operate on or perform best with grayscale images.\n"
                               "<b>When to use:</b> Early in the pipeline if color information is not necessary for the task. If color is needed for specific operations (e.g., color-based segmentation), apply grayscale conversion later in the pipeline or only when required for a specific step.",
                "param_explanations": {} # Grayscale conversion typically has no user-adjustable parameters
            },
            "Gaussian Blur": {
                "explanation": "Blurs an image by convolving it with a Gaussian kernel. This is a common method for reducing image noise and detail.\n\n"
                               "<b>Why:</b> To smooth out image noise, which can interfere with algorithms that are sensitive to high-frequency changes (like edge detectors or thresholding). Can also be used to suppress fine details before other processing steps.\n"
                               "<b>When to use:</b> Typically applied early in the pipeline, often before operations like Canny edge detection, adaptive thresholding, or contour finding.\n"
                               "<b>Interaction:</b> The size of the kernel (`ksize_w`, `ksize_h`) and the standard deviations (`sigmaX`, `sigmaY`) determine the extent of the blur. Larger values result in more blurring. Kernel dimensions must be positive and odd.",
                "param_explanations": {
                    "ksize_w": "<b>Kernel Width:</b> The width of the Gaussian kernel. Must be a positive, odd integer.",
                    "ksize_h": "<b>Kernel Height:</b> The height of the Gaussian kernel. Must be a positive, odd integer.",
                    "sigmaX": "<b>Sigma X:</b> The Gaussian kernel standard deviation in the horizontal (X) direction. If set to 0, it is automatically computed based on the kernel width (`ksize_w`).",
                    "sigmaY": "<b>Sigma Y:</b> The Gaussian kernel standard deviation in the vertical (Y) direction. If set to 0, it is automatically computed based on the kernel height (`ksize_h`). If both `sigmaX` and `sigmaY` are 0, they are both computed from the kernel dimensions. If `sigmaY` is 0 but `sigmaX` is not, `sigmaY` is set equal to `sigmaX`."
                }
            },
            "Bilateral Filter": {
                "explanation": "An edge-preserving smoothing filter. It reduces noise while attempting to keep the edges sharp, unlike conventional filters (like Gaussian) that blur edges.\n\n"
                               "<b>Why:</b> Useful for noise reduction in images where maintaining important structural edges is crucial, such as preparing images for object detection or segmentation without losing boundaries.\n"
                               "<b>When to use:</b> Typically applied early in the pipeline for cleaning the input image. It is generally more computationally expensive than simpler blurring filters like Gaussian or Median.\n"
                               "<b>Interaction:</b> It considers both the spatial distance between pixels (`sigmaSpace`) and the intensity/color difference (`sigmaColor`). Only pixels close in both aspects are averaged. A larger `d` value increases the neighborhood size, making the filter slower but potentially more effective over larger areas.",
                "param_explanations": {
                    "d": "<b>Diameter of Pixel Neighborhood:</b> Integer parameter. It defines the diameter of the circular neighborhood around each pixel over which the filter is applied. A non-positive value (<= 0) means it's calculated from `sigmaSpace`.\nTypical values range from 5 to 15. Larger values increase the area considered, leading to stronger filtering but significantly increasing computation time.",
                    "sigmaColor": "<b>Filter Sigma in Color Space:</b> Float parameter. A larger value means that pixels with colors/intensities that are significantly different from the center pixel will still be included in the averaging, provided they are spatially close (governed by `sigmaSpace`).\nIt controls the strength of color/intensity smoothing. Typical values are 50-150.",
                    "sigmaSpace": "<b>Filter Sigma in Coordinate Space:</b> Float parameter. A larger value means that pixels located farther away physically from the center pixel will still influence the averaging, provided their colors/intensities are close enough (governed by `sigmaColor`).\nIt controls the spatial extent of the filter. Typical values are 50-150."
                }
            },
            "CLAHE": {
                "explanation": "Contrast Limited Adaptive Histogram Equalization. This technique improves image contrast by applying histogram equalization to small, local regions (called tiles) of the image, rather than to the entire image globally.\n\n"
                               "<b>Why:</b> Particularly effective for enhancing local details and making features more visible in images with varying illumination conditions or where overall contrast is low in some areas but high in others. It overcomes the limitation of global histogram equalization which can over-enhance contrast in already bright areas.\n"
                               "<b>When to use:</b> Primarily applied to grayscale images. If used on a color image, it's common practice to convert it to a color space like LAB or HSV and apply CLAHE only to the luminance (L) or value (V) channel, then convert back.\n"
                               "<b>Interaction:</b> Can sometimes amplify noise, especially in uniform areas. Pre-filtering (e.g., Gaussian or Bilateral blur) might be beneficial. The `clipLimit` prevents over-enhancement of noise or artifacts.",
                "param_explanations": {
                    "clipLimit": "<b>Contrast Limit:</b> Float parameter. This is the threshold used for limiting contrast enhancement. Histogram bins within each tile that exceed this value are clipped before equalization. Higher values allow more aggressive contrast enhancement but can also amplify noise. A value of 0 effectively disables contrast limiting (standard Adaptive Histogram Equalization). Typical values are 2.0-5.0.",
                    "tileGridSize": "<b>Tile Grid Size:</b> Integer parameter. The image is divided into a grid of rectangular tiles of this size (e.g., 8 means an 8x8 grid). Histogram equalization is applied independently to each tile. This parameter determines the size of the local regions where contrast is adjusted.\nSmaller values increase the locality of the adjustment (more fine-grained), while larger values make it more similar to global equalization."
                }
            },
            "Adaptive Threshold": {
                "explanation": "Applies thresholding to a grayscale image where the threshold value is not fixed, but calculated dynamically for local neighborhoods of pixels.\n\n"
                               "<b>Why:</b> Ideal for segmenting foreground objects from the background in images where illumination conditions are uneven across the image, making a single global threshold ineffective.\n"
                               "<b>When to use:</b> Must be applied to a grayscale, 8-bit image. Often beneficial to apply a small amount of smoothing (e.g., Gaussian Blur) beforehand to reduce noise, which can otherwise cause artifacts in the thresholded image. The output is a binary image (pixels are either 0 or 255).",
                "param_explanations": {
                    "adaptiveMethod": "<b>Adaptive Method:</b> Specifies the algorithm used to calculate the local threshold value:\n  `Mean C`: The threshold is the mean value of the pixels in the `blockSize` x `blockSize` neighborhood, minus the constant `C`.\n  `Gaussian C`: The threshold is a weighted sum of neighborhood values using a Gaussian window, minus the constant `C`. This method is often less sensitive to noise than the mean method.",
                    "thresholdType": "<b>Threshold Type:</b> Determines the output pixel values relative to the threshold:\n  `Binary`: Pixels brighter than the threshold are set to the maximum value (255), and darker pixels are set to 0.\n  `Binary Inverse`: Pixels brighter than the threshold are set to 0, and darker pixels are set to the maximum value (255).",
                    "blockSize": "<b>Block Size:</b> The size of the square pixel neighborhood used to calculate the local threshold (e.g., 11 for an 11x11 block). This must be an odd integer greater than 1.",
                    "C": "<b>Constant C:</b> A constant value that is subtracted from the mean or weighted mean calculated by the adaptive method. It allows fine-tuning of the threshold. Positive values of C make the threshold lower (more pixels will be above the threshold), while negative values make it higher."
                }
            },
            "Morphological Operation": {
                "explanation": "Performs morphological transformations on images based on their shape. These operations are applied using a structuring element (kernel).\n\n"
                               "They are typically used on binary images, but can also be applied to grayscale images.\n"
                               "<b>Operations:</b>\n"
                               "  - <b>Erode:</b> Shrinks bright regions and expands dark regions. Useful for removing small white noise and disconnecting joined objects.\n"
                               "  - <b>Dilate:</b> Expands bright regions and shrinks dark regions. Useful for filling small holes, connecting broken objects, and highlighting features.\n"
                               "  - <b>Open:</b> An erosion followed by a dilation (Erode -> Dilate). Removes small bright spots (noise) and separates objects that are connected by a thin bridge.\n"
                               "  - <b>Close:</b> A dilation followed by an erosion (Dilate -> Erode). Fills small holes inside objects and connects nearby objects.\n"
                               "<b>When to use:</b> Commonly used after thresholding to clean up the resulting binary image or extract components. Also used for tasks like skeletonization or boundary extraction.",
                "param_explanations": {
                    "op_type": "<b>Operation Type:</b> Select the morphological operation to perform (Erode, Dilate, Open, Close).",
                    "kernel_shape": "<b>Kernel Shape:</b> Defines the shape of the structuring element used for the operation (Rectangle, Ellipse, Cross).",
                    "ksize_w": "<b>Kernel Width:</b> The width of the structuring element in pixels. Must be a positive integer.",
                    "ksize_h": "<b>Kernel Height:</b> The height of the structuring element in pixels. Must be a positive integer.",
                    "iterations": "<b>Iterations:</b> The number of times the morphological operation is applied sequentially. Applying an operation multiple times increases its effect."
                }
            },
            "Canny Edge": {
                "explanation": "A multi-stage algorithm to detect a wide range of edges in images. It is known for being relatively robust to noise and providing thin, connected edges.\n\n"
                               "<b>How:</b> Involves smoothing (Gaussian filter), finding intensity gradients, non-maximum suppression, and hysteresis thresholding.\n"
                               "<b>Why:</b> Provides a clear representation of object boundaries, which is fundamental for many computer vision tasks like shape analysis, object recognition, and segmentation.\n"
                               "<b>When to use:</b> Typically applied to a grayscale image, often after some form of smoothing (like Gaussian Blur) to reduce noise that could lead to spurious edges. The output is a binary image where white pixels represent detected edges.",
                "param_explanations": {
                    "threshold1": "<b>Lower Threshold:</b> The first (lower) threshold used for the hysteresis procedure. Any edge pixel with a gradient magnitude below this threshold is suppressed.",
                    "threshold2": "<b>Higher Threshold:</b> The second (higher) threshold used for the hysteresis procedure. Any edge pixel with a gradient magnitude above this threshold is considered a definite edge.\nEdges with gradient magnitudes between `threshold1` and `threshold2` are considered only if they are connected to a definite edge pixel.",
                    "apertureSize": "<b>Aperture Size:</b> The size of the Sobel kernel used internally to compute the image gradients. It can take values 3, 5, or 7.",
                    "L2gradient": "<b>L2 Gradient:</b> Boolean flag. If True, the more accurate L2 norm is used to calculate the gradient magnitude: sqrt((dI/dx)^2 + (dI/dy)^2). If False (default), the less accurate but faster L1 norm is used: |dI/dx| + |dI/dy|."
                }
            },
            "Hough Circles": {
                "explanation": "Detects circles in a grayscale image using the Hough Transform. It is designed to find circular shapes defined by edges.\n\n"
                               "<b>Why:</b> Specifically suited for identifying circular objects like bottle caps, coins, pupils, or wheels when their shape is well-defined by edges.\n"
                               "<b>When to use:</b> Should be applied to a grayscale, 8-bit image. Often beneficial to apply a median or Gaussian blur beforehand to reduce noise, which can otherwise lead to false circle detections. It draws the detected circles on a color copy of the input image.\n"
                               "<b>Interaction:</b> Requires careful tuning of parameters depending on the image content, expected circle sizes, and density of circles.",
                "param_explanations": {
                    "dp": "<b>DP (Inverse Accumulator Ratio):</b> Float parameter. Specifies the inverse ratio of the accumulator resolution to the image resolution. `dp=1` means the accumulator has the same resolution as the input image. `dp=2` means the accumulator has half the resolution. Decreasing `dp` increases accumulator resolution and accuracy, but increases computation and memory usage.",
                    "minDist": "<b>Min Distance:</b> Minimum distance allowed between the centers of any two detected circles. Setting this too low can result in detecting multiple circles for a single object. Setting it too high might miss some closely packed circles. A value around image_rows / 8 or image_rows / 16 is a common starting point.",
                    "param1": "<b>Canny Upper Threshold:</b> The higher threshold for the Canny edge detector that is internally used in the Hough gradient method. Edges with gradients above this are strong edges.",
                    "param2": "<b>Accumulator Threshold:</b> The accumulator threshold for the circle centers at the detection stage. A smaller value will detect more circles, including potentially false positives. A larger value is more selective.",
                    "minRadius": "<b>Min Radius:</b> Minimum radius (in pixels) of circles to search for. Circles smaller than this will be ignored. Set to 0 to ignore this constraint.",
                    "maxRadius": "<b>Max Radius:</b> Maximum radius (in pixels) of circles to search for. Circles larger than this will be ignored. Set to 0 to ignore this constraint or make it equal to image diagonal for infinity."
                }
            },
            "Find Contours": {
                "explanation": "Finds continuous curves (contours) which represent the boundaries of connected components in a binary image.\n\n"
                               "<b>Why:</b> Fundamental for object analysis, shape detection, and measurement. Once contours are found, you can analyze their properties (area, perimeter, shape, etc.) or draw them.\n"
                               "<b>When to use:</b> Designed to work on binary, single-channel 8-bit images (where pixels are either 0 or 255). Typically applied after thresholding (e.g., simple, adaptive, or Otsu's) or edge detection (like Canny). If the input is not binary, the operation attempts to apply Otsu's threshold internally as a preprocessing step.\n"
                               "<b>Interaction:</b> `mode` determines the hierarchy of the detected contours (how nested contours are represented). `method` determines how contour points are stored (e.g., storing all points or approximating with simpler shapes). This operation draws all found contours, optionally filtered by area. For more complex filtering, use the 'Filter Contours' operation *after* this one.",
                "param_explanations": {
                    "mode": "<b>Retrieval Mode:</b> Specifies how contours are retrieved and organized. Options: `external` (only outermost contours), `list` (all contours without hierarchy), `ccomp` (connected components hierarchy), `tree` (full hierarchy tree).",
                    "method": "<b>Approximation Method:</b> Specifies how contour points are approximated:\n  `simple` (compresses horizontal, vertical, and diagonal segments, storing only endpoints).\n  `none` (stores all contour points, resulting in more points for complex shapes).",
                    "min_area_filter": "<b>Min Area (for drawing):</b> Only contours with an area greater than or equal to this minimum value will be drawn. Set to 0.0 to disable minimum area filtering for visualization.",
                    "max_area_filter": "<b>Max Area (for drawing):</b> Only contours with an area less than or equal to this maximum value will be drawn. Set to 0.0 or a very large value to disable maximum area filtering for visualization."

                }
            },
            "Filter Contours": {
                "explanation": "Filters contours based on various geometric properties calculated from their points (area, perimeter, bounding box, etc.). This operation finds contours internally (expecting a binary or near-binary image input) and then filters them before drawing the filtered set.\n\n"
                               "<b>Why:</b> To select specific shapes or objects based on criteria beyond just simple area, allowing for more precise object isolation (e.g., finding rectangles within a set of arbitrary shapes).\n"
                               "<b>When to use:</b> Typically applied to a binary image (output of Canny or thresholding). It can replace a `Find Contours` step if you only need the *filtered* contours drawn.\n"
                               "<b>Interaction:</b> Parameters define the acceptable ranges for contour properties like area, aspect ratio (width / height of the bounding box), and circularity (a measure of how closely the contour resembles a perfect circle). The operation draws only the contours that satisfy *all* enabled filter criteria.",
                "param_explanations": {
                    "min_area": "<b>Min Area:</b> Minimum area (in pixels) a contour must have to pass the filter.",
                    "max_area": "<b>Max Area:</b> Maximum area (in pixels) a contour can have to pass the filter.",
                    "min_aspect_ratio": "<b>Min Aspect Ratio (W/H):</b> Minimum aspect ratio (width of bounding box / height of bounding box) a contour must have. Use for filtering elongated shapes.",
                    "max_aspect_ratio": "<b>Max Aspect Ratio (W/H):</b> Maximum aspect ratio (width of bounding box / height of bounding box) a contour can have. Use for filtering elongated shapes.",
                    "min_circularity": "<b>Min Circularity:</b> Minimum value for the circularity metric ((4 * pi * Area) / (Perimeter ^ 2)). A perfect circle has a circularity of 1. Values closer to 1 indicate shapes closer to a circle.",
                    "max_circularity": "<b>Max Circularity:</b> Maximum value for the circularity metric. Can be greater than 1 for irregular shapes with high perimeter relative to area (e.g., complex boundaries, noisy contours)."
                }
            },
    }
    # List of available operations with default parameters
    # Explicitly type the list and its contents. This definition seems correct.
    available_op_configs: List[AvailableOpConfigEntry] = [
        {"display_name": "Undistort", "op_type_name": "Undistort", "params": {"camera_matrix_str": {"type": "text", "value": "1000,0,320;0,1000,240;0,0,1"}, "dist_coeffs_str": {"type": "text", "value": "0,0,0,0,0"}, "alpha": {"type": "float", "value": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "decimals": 2}}},
        {"display_name": "Resize", "op_type_name": "Resize", "params": {"width": {"type": "int", "value": 640, "min": 100, "max": 3840, "step": 10}, "height": {"type": "int", "value": 480, "min": 100, "max": 2160, "step": 10}, "interpolation": {"type": "choice", "value": "linear", "choices": {"Nearest": "nearest", "Linear": "linear", "Area": "area", "Cubic": "cubic", "Lanczos4": "lanczos4"}}}},
        {"display_name": "Grayscale", "op_type_name": "Grayscale", "params": {}},
        {"display_name": "Gaussian Blur", "op_type_name": "Gaussian Blur", "params": {"ksize_w": {"type": "int", "value": 5, "min": 1, "max": 51, "step": 2}, "ksize_h": {"type": "int", "value": 5, "min": 1, "max": 51, "step": 2}, "sigmaX": {"type": "float", "value": 0.0, "min": 0.0, "max": 50.0, "step": 0.1, "decimals": 1}, "sigmaY": {"type": "float", "value": 0.0, "min": 0.0, "max": 50.0, "step": 0.1, "decimals": 1}}},
        {"display_name": "Bilateral Filter", "op_type_name": "Bilateral Filter", "params": {"d": {"type": "int", "value": 9, "min": 1, "max": 25, "step": 2}, "sigmaColor": {"type": "float", "value": 75.0, "min": 1.0, "max": 250.0, "step": 1.0, "decimals":1}, "sigmaSpace": {"type": "float", "value": 75.0, "min": 1.0, "max": 250.0, "step": 1.0, "decimals":1}}},
        {"display_name": "CLAHE", "op_type_name": "CLAHE", "params": {"clipLimit": {"type": "float", "value": 2.0, "min": 0.1, "max": 40.0, "step": 0.1, "decimals":1}, "tileGridSize": {"type": "int", "value": 8, "min": 2, "max": 32, "step": 1}}},
        {"display_name": "Canny Edge", "op_type_name": "Canny Edge", "params": {"threshold1": {"type": "float", "value": 50.0, "min": 0.0, "max": 500.0, "step": 1.0, "decimals": 0}, "threshold2": {"type": "float", "value": 150.0, "min": 0.0, "max": 500.0, "step": 1.0, "decimals": 0}, "apertureSize": {"type": "choice", "value": 3, "choices": {"3x3": 3, "5x5": 5, "7x7": 7}}, "L2gradient": {"type": "bool", "value": False}}},
        {"display_name": "Adaptive Threshold", "op_type_name": "Adaptive Threshold", "params": {"adaptiveMethod": {"type": "choice", "value": 1, "choices": {"Mean C": 0, "Gaussian C": 1}}, "thresholdType": {"type": "choice", "value": 1, "choices": {"Binary": 0, "Binary Inverse": 1}}, "blockSize": {"type": "int", "value": 45, "min": 3, "max": 101, "step": 2}, "C": {"type": "float", "value": 15.0, "min": -50.0, "max": 50.0, "step": 1.0, "decimals": 0}}},
        {"display_name": "Morphological Operation", "op_type_name": "Morphological Operation", "params": {"op_type": {"type": "choice", "value": "erode", "choices": {"Erode": "erode", "Dilate": "dilate", "Open": "open", "Close": "close"}}, "kernel_shape": {"type": "choice", "value": "rect", "choices": {"Rectangle": "rect", "Ellipse": "ellipse", "Cross": "cross"}}, "ksize_w": {"type": "int", "value": 5, "min": 1, "max": 51, "step": 1}, "ksize_h": {"type": "int", "value": 5, "min": 1, "max": 51, "step": 1}, "iterations": {"type": "int", "value": 1, "min": 1, "max": 10, "step": 1}}},
        {"display_name": "Hough Circles", "op_type_name": "Hough Circles", "params": {"dp": {"type": "float", "value": 1.2, "min": 1.0, "max": 5.0, "step": 0.1, "decimals": 1}, "minDist": {"type": "float", "value": 20.0, "min": 1.0, "max": 1000.0, "step": 1.0, "decimals": 0, "tooltip":"Minimum distance between centers of detected circles. Typically image_rows / N"}, "param1": {"type": "float", "value": 100.0, "min": 1.0, "max": 300.0, "step": 1.0, "decimals": 0, "tooltip":"Higher threshold for Canny edge detector (internal)."}, "param2": {"type": "float", "value": 30.0, "min": 1.0, "max": 200.0, "step": 1.0, "decimals": 0, "tooltip":"Accumulator threshold for circle detection. Smaller = more false circles."}, "minRadius": {"type": "int", "value": 10, "min": 0, "max": 500, "step": 1}, "maxRadius": {"type": "int", "value": 100, "min": 0, "max": 1000, "step": 1}}},
        {"display_name": "Find Contours", "op_type_name": "Find Contours", "params": {"mode": {"type": "choice", "value": "external", "choices": {"External": "external", "List": "list", "CComp": "ccomp", "Tree": "tree"}}, "method": {"type": "choice", "value": "simple", "choices": {"Simple Approx.": "simple", "None (All Points)": "none"}}, "min_area_filter": {"type": "float", "value": 0.0, "min":0.0, "max":1000000.0, "step":10.0, "decimals":0, "tooltip": "Minimum contour area to keep (0 for no filter)."}, "max_area_filter": {"type": "float", "value": 0.0, "min":0.0, "max":5000000.0, "step":100.0, "decimals":0, "tooltip": "Maximum contour area to keep (0 for no max area filter, effectively infinity)."}}},
        {"display_name": "Filter Contours", "op_type_name": "Filter Contours", "params": {"min_area": {"type": "float", "value": 100.0, "min": 0.0, "max": 1000000.0, "step": 10.0, "decimals":0}, "max_area": {"type": "float", "value": 50000.0, "min": 0.0, "max": 5000000.0, "step": 100.0, "decimals":0}, "min_aspect_ratio": {"type": "float", "value": 0.1, "min": 0.01, "max": 100.0, "step": 0.05, "decimals":2}, "max_aspect_ratio": {"type": "float", "value": 100.0, "min": 0.1, "max": 1000.0, "step": 0.1, "decimals":2}, "min_circularity": {"type": "float", "value": 0.1, "min": 0.0, "max": 2.0, "step": 0.01, "decimals":2}, "max_circularity": {"type": "float", "value": 2.0, "min": 0.0, "max": 10.0, "step": 0.01, "decimals":2}}},
    ]

    # Mapping operation type names to their processing functions
    # Explicitly type the dictionary
    processing_functions: Dict[str, ProcessingFunction] = {
        "Undistort": apply_undistort, "Resize": apply_resize,
        "Grayscale": apply_grayscale, "Bilateral Filter": apply_bilateral_filter,
        "CLAHE": apply_clahe, "Adaptive Threshold": apply_adaptive_threshold,
        "Gaussian Blur": apply_gaussian_blur,
        "Morphological Operation": apply_morphological_op, "Canny Edge": apply_canny_edge,
        "Hough Circles": apply_hough_circles, "Find Contours": apply_find_contours,
        "Filter Contours": apply_filter_contours,
    }


    # Declare UI elements explicitly for better type tracking.
    # They are Optional until init_ui is called, although usually non-None after.
    central_widget: QWidget
    main_layout: QHBoxLayout
    source_type_combo: QComboBox
    camera_id_spinbox: QSpinBox
    video_file_edit: QLineEdit
    browse_video_button: QPushButton
    rtsp_url_edit: QLineEdit
    connect_source_button: QPushButton
    operations_list_widget: QListWidget
    add_op_button: QPushButton
    remove_op_button: QPushButton
    save_pipeline_button: QPushButton
    load_pipeline_button: QPushButton
    parameter_panel_scroll_area: QScrollArea
    parameter_panel_title_label: QLabel
    parameter_panel_op_name_label: QLabel
    video_display_label: QLabel
    original_video_label: QLabel
    stage_input_video_label: QLabel
    stage_output_video_label: QLabel


    def __init__(self, video_source_arg: Union[int, str] = 0, parent: Optional[QWidget] = None):
        super().__init__(parent)
        # Validate video_source_arg type - Diagnostic reportUnnecessaryIsInstance removed.
        # The check `isinstance(video_source_arg, Union[int, str])` is redundant as the type is in the signature.
        # Removing the redundant check.

        self.video_source = video_source_arg # Assign directly if type is correct

        self.setWindowTitle("Image Processing Pipeline Tuner")
        # Set initial window size and position
        # Diagnostic reportUnknownMemberType on setGeometry stub addressed by ignore
        self.setGeometry(50, 50, 1700, 900) # type: ignore[reportUnknownMemberType]

        # Initialize state variables with their declared types
        self.current_cv_frame = None
        self.original_frame_for_display = None
        self.selected_stage_input_frame = None
        self.selected_stage_output_frame = None
        self.operations_data = []
        self.video_thread = None
        self.first_frame_received = False
        # Initialize the list for managing signal connections explicitly
        self._connections = [] # Type is already declared above

        # Configuration data initialized above instance variables

        # Initialize the User Interface
        self.init_ui()
        # Populate the initial pipeline operations (e.g., for a demo/bottles)
        self.populate_initial_operations_for_bottles()

    def init_ui(self) -> None:
        """Sets up the main application window UI."""
        # Diagnostic reportUnknownMemberType on setStyleSheet stub addressed by ignore
        self.setStyleSheet(self.get_stylesheet()) # Apply custom stylesheet # type: ignore[reportUnknownMemberType]

        self.central_widget = QWidget(); self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left panel for source and pipeline controls
        left_panel_widget = QWidget(); left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_widget.setFixedWidth(350) # setFixedWidth stub

        # Video Source Groupbox
        source_groupbox = QGroupBox("Video Source"); source_layout = QGridLayout(source_groupbox) # Set layout directly on groupbox
        self.source_type_combo = QComboBox()
        # Standard PyQt method. Add ignore.
        self.source_type_combo.addItems(["Camera ID", "Video File", "RTSP Stream"]) # type: ignore[reportUnknownMemberType]
        # Standard connection. Add ignore.
        self._connections.append(self.source_type_combo.currentIndexChanged.connect(self.update_source_input_visibility)) # type: ignore[reportUnknownMemberType]

        source_layout.addWidget(QLabel("Type:"), 0, 0); source_layout.addWidget(self.source_type_combo, 0, 1, 1, 2)

        # Camera ID input
        self.camera_id_spinbox = QSpinBox(); self.camera_id_spinbox.setRange(0, 99)
        # Set initial value based on self.video_source if it's an int
        if isinstance(self.video_source, int):
             self.camera_id_spinbox.setValue(self.video_source) # setValue stub
        else:
             self.camera_id_spinbox.setValue(0) # Default value if source is not int # setValue stub

        # Video File input
        self.video_file_edit = QLineEdit(); self.video_file_edit.setPlaceholderText("Path to video file") # setPlaceholderText stub
        self.browse_video_button = QPushButton("Browse...")
        # Standard connection. Add ignore.
        self._connections.append(self.browse_video_button.clicked.connect(self.browse_for_video_file)) # type: ignore[reportUnknownMemberType]

        # RTSP URL input
        self.rtsp_url_edit = QLineEdit(); self.rtsp_url_edit.setPlaceholderText("rtsp://user:pass@ip:port/...") # setPlaceholderText stub

        # Add source input widgets to layout (visibility controlled by update_source_input_visibility)
        source_layout.addWidget(QLabel("ID/Path:"), 1, 0)
        source_layout.addWidget(self.camera_id_spinbox, 1, 1, 1, 2) # Span 2 columns
        source_layout.addWidget(self.video_file_edit, 1, 1)
        source_layout.addWidget(self.browse_video_button, 1, 2)
        source_layout.addWidget(self.rtsp_url_edit, 1, 1, 1, 2) # Span 2 columns

        # Set initial value for file/rtsp if applicable
        if isinstance(self.video_source, str):
             if self.video_source.lower().startswith("rtsp://"):
                  # Diagnostic reportUnknownMemberType on setCurrentText stub addressed by ignore
                  self.rtsp_url_edit.setText(self.video_source) # setText stub
                  self.source_type_combo.setCurrentText("RTSP Stream") # type: ignore[reportUnknownMemberType]
             elif os.path.exists(self.video_source):
                  # Diagnostic reportUnknownMemberType on setCurrentText stub addressed by ignore
                  self.video_file_edit.setText(self.video_source) # setText stub
                  self.source_type_combo.setCurrentText("Video File") # type: ignore[reportUnknownMemberType]
             else:
                  print(f"Warning: Initial video source string '{self.video_source}' is not a valid file path or RTSP URL. Defaulting to camera ID 0.")
                  self.video_source = 0 # Fallback to camera 0
                  # Diagnostic reportUnknownMemberType on setCurrentText stub addressed by ignore
                  self.source_type_combo.setCurrentText("Camera ID") # type: ignore[reportUnknownMemberType]


        # Update initial visibility based on default source type set above
        self.update_source_input_visibility()

        # Connect/Stop button
        self.connect_source_button = QPushButton("Connect Source")
        # Standard connection. Add ignore.
        self._connections.append(self.connect_source_button.clicked.connect(self.connect_source_slot)) # type: ignore[reportUnknownMemberType]
        source_layout.addWidget(self.connect_source_button, 2, 0, 1, 3) # Span 3 columns

        left_panel_layout.addWidget(source_groupbox)

        # Processing Pipeline List Groupbox
        ops_list_groupbox = QGroupBox("Processing Pipeline"); ops_list_groupbox_layout = QVBoxLayout(ops_list_groupbox)
        ops_list_groupbox_layout.addWidget(QLabel("<b>Steps</b> (Drag to Reorder):"))
        self.operations_list_widget = QListWidget()
        # Diagnostic reportUnknownMemberType on setDragDropMode stub addressed by ignore
        self.operations_list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove) # type: ignore[reportUnknownMemberType]
        # Diagnostic reportUnknownMemberType on setSelectionMode stub addressed by ignore
        self.operations_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection) # type: ignore[reportUnknownMemberType]
        list_font = QFont();
        # Diagnostic reportUnknownMemberType on setPointSize stub addressed by ignore
        list_font.setPointSize(10); # type: ignore[reportUnknownMemberType]
        # Diagnostic reportUnknownMemberType on setFont stub addressed by ignore
        self.operations_list_widget.setFont(list_font) # type: ignore[reportUnknownMemberType]
        ops_list_groupbox_layout.addWidget(self.operations_list_widget)

        # Connect the rowsMoved signal to update internal data structure
        # Diagnostic reportUnknownMemberType on model stub addressed by ignore
        model = self.operations_list_widget.model() # type: ignore[reportUnknownMemberType]
        if model:
             # Replace lambda with typed method to resolve reportUnknownLambdaType
             # Diagnostic reportUnknownMemberType on rowsMoved.connect stub addressed by ignore
             self._connections.append(model.rowsMoved.connect(self._handle_list_widget_rows_moved)) # type: ignore[reportUnknownMemberType]


        # Operation management buttons (Add/Remove)
        op_management_layout = QHBoxLayout()
        self.add_op_button = QPushButton("➕ Add Op")
        # Standard connection. Add ignore.
        self._connections.append(self.add_op_button.clicked.connect(self.prompt_add_operation)) # type: ignore[reportUnknownMemberType]
        op_management_layout.addWidget(self.add_op_button)
        self.remove_op_button = QPushButton("➖ Remove Op")
        # Standard connection. Add ignore.
        self._connections.append(self.remove_op_button.clicked.connect(self.remove_selected_operation)) # type: ignore[reportUnknownMemberType]
        op_management_layout.addWidget(self.remove_op_button)
        ops_list_groupbox_layout.addLayout(op_management_layout)

        left_panel_layout.addWidget(ops_list_groupbox)

        # Pipeline File Management Groupbox
        pipeline_io_groupbox = QGroupBox("Pipeline Management"); pipeline_io_layout = QHBoxLayout(pipeline_io_groupbox)
        self.save_pipeline_button = QPushButton("💾 Save Pipeline")
        # Standard connection. Add ignore.
        self._connections.append(self.save_pipeline_button.clicked.connect(self.save_pipeline_to_file)) # type: ignore[reportUnknownMemberType]
        pipeline_io_layout.addWidget(self.save_pipeline_button)
        self.load_pipeline_button = QPushButton("📂 Load Pipeline")
        # Standard connection. Add ignore.
        self._connections.append(self.load_pipeline_button.clicked.connect(self.load_pipeline_from_file)) # type: ignore[reportUnknownMemberType]
        pipeline_io_layout.addWidget(self.load_pipeline_button)

        left_panel_layout.addWidget(pipeline_io_groupbox)

        # Diagnostic reportUnknownMemberType on addStretch stub addressed by ignore
        left_panel_layout.addStretch(1) # Add stretch to push everything to the top # type: ignore[reportUnknownMemberType]
        self.main_layout.addWidget(left_panel_widget) # Add left panel to main layout

        # Parameter Panel (Scrollable)
        self.parameter_panel_scroll_area = QScrollArea();
        # Diagnostic reportUnknownMemberType on setWidgetResizable stub addressed by ignore
        self.parameter_panel_scroll_area.setWidgetResizable(True) # type: ignore[reportUnknownMemberType]
        # Diagnostic reportUnknownMemberType on setFixedWidth stub addressed by ignore
        self.parameter_panel_scroll_area.setFixedWidth(420) # type: ignore[reportUnknownMemberType]
        # Diagnostic reportUnknownMemberType on setStyleSheet stub addressed by ignore
        self.parameter_panel_scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #3a3a3a; }") # Style scroll area # type: ignore[reportUnknownMemberType]

        parameter_panel_content_widget = QWidget()
        self.parameter_widgets_area_layout = QVBoxLayout(parameter_panel_content_widget)
        self.parameter_widgets_area_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Align content to the top

        # Labels for the parameter panel header
        self.parameter_panel_title_label = QLabel("<b>Parameters & Info:</b>")
        self.parameter_panel_op_name_label = QLabel("<i>No operation selected</i>")
        # Diagnostic reportUnknownMemberType on setStyleSheet stub addressed by ignore
        self.parameter_panel_op_name_label.setStyleSheet("font-style: italic; color: #aaa; margin-bottom: 5px; font-size: 11pt;") # type: ignore[reportUnknownMemberType]

        self.parameter_widgets_area_layout.addWidget(self.parameter_panel_title_label)
        self.parameter_widgets_area_layout.addWidget(self.parameter_panel_op_name_label)

        # Label for the operation explanation
        self.operation_explanation_label = QLabel("<i>Select an operation to see details.</i>")
        # Diagnostic reportUnknownMemberType on setWordWrap stub addressed by ignore
        self.operation_explanation_label.setWordWrap(True); # type: ignore[reportUnknownMemberType]
        self.operation_explanation_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Diagnostic reportUnknownMemberType on setStyleSheet stub addressed by ignore
        self.operation_explanation_label.setStyleSheet("font-size: 9pt; color: #c8c8c8; margin-top: 5px; margin-bottom:10px; border: 1px solid #404040; padding: 8px; background-color: #303030; border-radius: 4px;") # type: ignore[reportUnknownMemberType]
        self.parameter_widgets_area_layout.addWidget(self.operation_explanation_label)

        # Set the content widget for the scroll area
        # Diagnostic reportUnknownMemberType on setWidget stub addressed by ignore
        self.parameter_panel_scroll_area.setWidget(parameter_panel_content_widget) # type: ignore[reportUnknownMemberType]
        self.main_layout.addWidget(self.parameter_panel_scroll_area) # Add scroll area to main layout

        # Right panel for video displays
        video_views_panel = QWidget(); video_views_layout = QVBoxLayout(video_views_panel)

        # Final Output display label
        self.video_display_label = QLabel("Final Output"); self.video_display_label.setObjectName("videoLabel") # setObjectName stub
        self.video_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.video_display_label.setMinimumSize(480, 360) # setMinimumSize stub
        # Size policy allows label to grow/shrink but prioritizes maintaining aspect ratio for content
        self.video_display_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) # setSizePolicy stub
        video_views_layout.addWidget(QLabel("<b>Final Output:</b>"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.video_display_label.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Plain) # setFrameStyle stub
        video_views_layout.addWidget(self.video_display_label, 2) # Give this label a stretch factor of 2

        # Intermediate views layout (Original, Stage Input, Stage Output)
        intermediate_views_layout = QHBoxLayout()

        # Original Input view
        original_view_container = QWidget(); original_view_Vlayout = QVBoxLayout(original_view_container)
        self.original_video_label = QLabel("Original"); self.original_video_label.setObjectName("originalVideoLabel") # setObjectName stub
        self.original_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.original_video_label.setMinimumSize(240,180) # setMinimumSize stub
        self.original_video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) # setSizePolicy stub
        original_view_Vlayout.addWidget(QLabel("Original Input:"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.original_video_label.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Plain) # setFrameStyle stub
        original_view_Vlayout.addWidget(self.original_video_label); intermediate_views_layout.addWidget(original_view_container)

        # Stage Input view
        stage_input_view_container = QWidget(); stage_input_view_Vlayout = QVBoxLayout(stage_input_view_container)
        self.stage_input_video_label = QLabel("Input to Stage"); self.stage_input_video_label.setObjectName("stageInputVideoLabel") # setObjectName stub
        self.stage_input_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.stage_input_video_label.setMinimumSize(240,180) # setMinimumSize stub
        self.stage_input_video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) # setSizePolicy stub
        stage_input_view_Vlayout.addWidget(QLabel("Input to Selected Stage:"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.stage_input_video_label.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Plain) # setFrameStyle stub
        stage_input_view_Vlayout.addWidget(self.stage_input_video_label); intermediate_views_layout.addWidget(stage_input_view_container)

        # Stage Output view
        stage_output_view_container = QWidget(); stage_output_view_Vlayout = QVBoxLayout(stage_output_view_container)
        self.stage_output_video_label = QLabel("Output of Stage"); self.stage_output_video_label.setObjectName("stageOutputVideoLabel") # setObjectName stub
        self.stage_output_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.stage_output_video_label.setMinimumSize(240,180) # setMinimumSize stub
        self.stage_output_video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) # setSizePolicy stub
        stage_output_view_Vlayout.addWidget(QLabel("Output of Selected Stage:"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.stage_output_video_label.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Plain) # setFrameStyle stub
        stage_output_view_Vlayout.addWidget(self.stage_output_video_label); intermediate_views_layout.addWidget(stage_output_view_container)

        # Add intermediate views layout to the video panel layout
        video_views_layout.addLayout(intermediate_views_layout, 1) # Give this layout a stretch factor of 1
        self.main_layout.addWidget(video_views_panel, 1) # Add video panel to main layout with stretch factor 1

        # Connect signal for list widget item selection changes
        # QListWidget.currentItemChanged signal signature: current, previous (QListWidgetItem | None)
        # Explicitly hint lambda parameters using cast and ignore lambda type issues.
        # Diagnostic reportUnknownMemberType on currentItemChanged.connect stub addressed by ignore
        # Diagnostic reportUnknownLambdaType on lambda parameters addressed by ignore
        self._connections.append(self.operations_list_widget.currentItemChanged.connect( # type: ignore[reportUnknownMemberType]
            lambda current, previous: self.on_operation_selected(cast(Optional[QListWidgetItem], current), cast(Optional[QListWidgetItem], previous)) # type: ignore[reportUnknownLambdaType]
        ))

    def update_source_input_visibility(self, _index: Optional[int] = None) -> None:
        """Adjusts visibility of source input widgets based on selected source type."""
        # Ensure _index is not used (it's emitted by the signal but not needed here)
        _ = _index

        # Diagnostic reportUnknownMemberType on currentText stub addressed by ignore
        source_type = self.source_type_combo.currentText() # type: ignore[reportUnknownMemberType]
        # Set visibility for each set of widgets based on the selected source type string.
        # Use string comparison directly.
        is_camera = source_type == "Camera ID"
        is_file = source_type == "Video File"
        is_rtsp = source_type == "RTSP Stream"

        # Diagnostic reportUnknownMemberType on setVisible stub addressed by ignore
        self.camera_id_spinbox.setVisible(is_camera) # type: ignore[reportUnknownMemberType]
        # Diagnostic reportUnknownMemberType on setVisible stub addressed by ignore
        self.video_file_edit.setVisible(is_file) # type: ignore[reportUnknownMemberType]
        # Diagnostic reportUnknownMemberType on setVisible stub addressed by ignore
        self.browse_video_button.setVisible(is_file) # type: ignore[reportUnknownMemberType]
        # Diagnostic reportUnknownMemberType on setVisible stub addressed by ignore
        self.rtsp_url_edit.setVisible(is_rtsp) # type: ignore[reportUnknownMemberType]

    def browse_for_video_file(self) -> None:
        """Opens a file dialog for the user to select a video file."""
        # QFileDialog.getOpenFileName returns a tuple: (filePath: str, filter: str)
        # Specify the parent widget (self), dialog title, default directory, and file filters.
        # Standard method. Add ignore.
        filePath, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)") # type: ignore[reportUnknownMemberType]
        # If a file path was selected and is not an empty string
        if filePath:
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            self.video_file_edit.setText(filePath) # type: ignore[reportUnknownMemberType]

    def connect_source_slot(self) -> None:
        """Slot to handle Connect/Stop Stream button click."""
        # Diagnostic reportUnknownMemberType on text stub addressed by ignore
        if self.connect_source_button.text() == "Stop Stream": # type: ignore[reportUnknownMemberType]
            # Check if the video thread exists and is currently running
            # Check self.video_thread is not None before accessing methods. Keep check, ignore diagnostic.
            # Diagnostic reportUnknownMemberType on isRunning stub addressed by ignore
            if self.video_thread is not None and self.video_thread.isRunning(): # type: ignore[reportUnnecessaryComparison, reportUnknownMemberType]
                self.video_thread.stop() # Signal thread to stop
                # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
                self.connect_source_button.setText("Stopping...") # Update button text # type: ignore[reportUnknownMemberType]
                # Diagnostic reportUnknownMemberType on setEnabled stub addressed by ignore
                self.connect_source_button.setEnabled(False) # Disable button while stopping # type: ignore[reportUnknownMemberType]
            else: # Handle edge case where thread isn't running but button says Stop
                print("Warning: Connect button state out of sync with video thread.")
                # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
                self.connect_source_button.setText("Connect Source") # type: ignore[reportUnknownMemberType]
                # Diagnostic reportUnknownMemberType on setEnabled stub addressed by ignore
                self.connect_source_button.setEnabled(True) # type: ignore[reportUnknownMemberType]
            return # Exit the slot after handling stop action

        # If button text is "Connect Source", start a new thread

        # Clean up any existing thread before starting a new one
        # Check self.video_thread is not None before accessing methods. Keep check, ignore diagnostic.
        if self.video_thread is not None: # type: ignore[reportUnnecessaryComparison]
             # Diagnostic reportUnknownMemberType on isRunning stub addressed by ignore
             if self.video_thread.isRunning(): # type: ignore[reportUnknownMemberType]
                 self.video_thread.stop()
                 # Wait a bit for the thread to finish gracefully (e.g., 1 second)
                 print("Waiting for previous video thread to finish...")
                 # Check return value of wait if needed, or increase timeout.
                 # Using 1000ms timeout. Standard method. Add ignore.
                 # Diagnostic reportUnknownMemberType on wait stub addressed by ignore
                 if not self.video_thread.wait(1000): # type: ignore[reportUnknownMemberType]
                     print("Warning: Old video thread did not stop within timeout.")
             # Mark the thread object for deletion by Qt's event loop.
             # This also implicitly deletes child objects like emitters, disconnecting signals.
             # Diagnostic reportUnknownMemberType on deleteLater stub addressed by ignore
             self.video_thread.deleteLater() # type: ignore[reportUnknownMemberType]
             self.video_thread = None # Clear the reference

        # Determine the video source based on the UI selection
        # Diagnostic reportUnknownMemberType on currentText stub addressed by ignore
        source_type = self.source_type_combo.currentText() # type: ignore[reportUnknownMemberType]
        current_video_source_ui: Union[int, str] # Declare type for clarity
        if source_type == "Camera ID":
            # Diagnostic reportUnknownMemberType on value stub addressed by ignore
            current_video_source_ui = self.camera_id_spinbox.value() # QSpinBox.value() returns int # type: ignore[reportUnknownMemberType]
        elif source_type == "Video File":
            # Diagnostic reportUnknownMemberType on text stub addressed by ignore
            current_video_source_ui = self.video_file_edit.text() # QLineEdit.text() returns str # type: ignore[reportUnknownMemberType]
            # Validate file existence
            if not current_video_source_ui or not os.path.exists(current_video_source_ui):
                # Diagnostic reportUnknownMemberType on warning stub addressed by ignore
                QMessageBox.warning(self, "Source Error", "Video file path is invalid or file does not exist."); return # type: ignore[reportUnknownMemberType]
        elif source_type == "RTSP Stream":
            # Diagnostic reportUnknownMemberType on text stub addressed by ignore
            current_video_source_ui = self.rtsp_url_edit.text() # QLineEdit.text() returns str # type: ignore[reportUnknownMemberType]
            # Basic validation for RTSP URL format
            if not current_video_source_ui.lower().startswith("rtsp://"):
                # Diagnostic reportUnknownMemberType on warning stub addressed by ignore
                QMessageBox.warning(self, "Source Error", "Invalid RTSP URL format."); return # type: ignore[reportUnknownMemberType]
        else: # Should not be reached if combo box is exhaustive
            # Diagnostic reportUnknownMemberType on critical stub addressed by ignore
            QMessageBox.critical(self, "Internal Error", "Unknown source type selected."); return # type: ignore[reportUnknownMemberType]

        # Set the main video source property of the QMainWindow
        self.video_source = current_video_source_ui
        self.first_frame_received = False # Reset flag for new connection

        # Update video display labels to indicate connecting state
        # Diagnostic reportUnknownMemberType on clear/setText stub addressed by ignore
        self.original_video_label.clear(); self.original_video_label.setText("Connecting...") # type: ignore[reportUnknownMemberType]
        self.video_display_label.clear(); self.video_display_label.setText("Connecting...") # type: ignore[reportUnknownMemberType]
        self.stage_input_video_label.clear(); self.stage_input_video_label.setText("N/A") # type: ignore[reportUnknownMemberType]
        self.stage_output_video_label.clear(); self.stage_output_video_label.setText("N/A") # type: ignore[reportUnknownMemberType]

        # Clear current frame data
        self.current_cv_frame = None
        self.original_frame_for_display = None
        # Clear stage views data as well
        self.selected_stage_input_frame = None
        self.selected_stage_output_frame = None


        # Start the video capture thread
        self.start_video_thread()

    def start_video_thread(self) -> None:
        """Creates and starts the video capture thread."""
        # Prevent starting if a thread is already running
        # Check self.video_thread is not None before accessing methods. Keep check, ignore diagnostic.
        # Diagnostic reportUnknownMemberType on isRunning stub addressed by ignore
        if self.video_thread is not None and self.video_thread.isRunning(): # type: ignore[reportUnnecessaryComparison, reportUnknownMemberType]
            print("Video thread is already running.")
            return

        # Create a new VideoThread instance with the selected source
        # Pass self as the parent so signals are correctly handled in the main thread
        self.video_thread = VideoThread(video_source=self.video_source, parent=self)

        # Connect signals from the thread to slots in the main window
        # Use self._connections to manage these connections for explicit disconnection on close.
        # The signal `new_frame_signal` emits `object` for ndarray compatibility with PyQt.
        # The receiving slot `update_video_frame_slot` takes `object`. Standard connection. Add ignore.
        # Diagnostic reportUnknownMemberType on connect stub addressed by ignore
        self._connections.append(self.video_thread.new_frame_signal.connect(self.update_video_frame_slot)) # type: ignore[reportUnknownMemberType]
        # The signal `error_signal` emits `str`. Standard connection. Add ignore.
        # Diagnostic reportUnknownMemberType on connect stub addressed by ignore
        self._connections.append(self.video_thread.error_signal.connect(self.show_video_error_slot)) # type: ignore[reportUnknownMemberType]
        # The signal `finished_signal` emits no arguments. Standard connection. Add ignore.
        # Diagnostic reportUnknownMemberType on connect stub addressed by ignore
        self._connections.append(self.video_thread.finished_signal.connect(self.on_video_thread_finished)) # type: ignore[reportUnknownMemberType]

        # Start the thread execution
        # Diagnostic reportUnknownMemberType on start stub addressed by ignore
        self.video_thread.start() # type: ignore[reportUnknownMemberType]

        # Update UI state while connecting
        # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
        self.connect_source_button.setText("Connecting...") # type: ignore[reportUnknownMemberType]
        # Disable button until connection status is known (first frame or error)
        # Diagnostic reportUnknownMemberType on setEnabled stub addressed by ignore
        self.connect_source_button.setEnabled(False) # type: ignore[reportUnknownMemberType]

    def on_video_thread_finished(self) -> None:
        """Slot called when the video thread finishes."""
        print("Video thread finished.")

        # Check if thread object exists before accessing it
        # Check self.video_thread is not None. Keep check, ignore diagnostic.
        if self.video_thread is not None: # type: ignore[reportUnnecessaryComparison]
            # Ensure thread is stopped (should be if finished_signal emitted) and wait briefly.
            # Diagnostic reportUnknownMemberType on isRunning stub addressed by ignore
            if self.video_thread.isRunning(): # type: ignore[reportUnknownMemberType]
                print("Warning: Video thread finished signal emitted, but thread still reports running.")
                self.video_thread.stop() # Ensure stop flag is set
                # Add a final brief wait. Standard method. Add ignore.
                # Diagnostic reportUnknownMemberType on wait stub addressed by ignore
                if not self.video_thread.wait(100): # type: ignore[reportUnknownMemberType]
                    print("Warning: Video thread did not terminate after stop command.")

            # The thread object will be deleted when the parent window is deleted due to parentage.
            # Manual disconnection happens in closeEvent.
            self.video_thread = None # Clear reference

        # Restore Connect button state
        # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
        self.connect_source_button.setText("Connect Source") # type: ignore[reportUnknownMemberType]
        # Diagnostic reportUnknownMemberType on setEnabled stub addressed by ignore
        self.connect_source_button.setEnabled(True) # type: ignore[reportUnknownMemberType]

        # Update labels if no frame was ever received (e.g., instant error or empty file)
        if not self.first_frame_received :
             # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
             self.original_video_label.setText("Source Ended/Error") # type: ignore[reportUnknownMemberType]
             # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
             self.video_display_label.setText("Source Ended/Error") # type: ignore[reportUnknownMemberType]
             # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
             self.stage_input_video_label.setText("N/A") # type: ignore[reportUnknownMemberType]
             # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
             self.stage_output_video_label.setText("N/A") # type: ignore[reportUnknownMemberType]

    def show_video_error_slot(self, error_message: str) -> None:
        """Slot to display video thread error messages."""
        print(f"Video Error: {error_message}")
        # Update the main display label with the error message and style it red.
        # Use HTML for rich text formatting in the label
        # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
        self.video_display_label.setText(f"<html><body style='text-align:center;'><p>Video Error:</p><p>{error_message}</p></body></html>") # type: ignore[reportUnknownMemberType]
        # Apply a specific stylesheet for error state
        # Diagnostic reportUnknownMemberType on setStyleSheet stub addressed by ignore
        self.video_display_label.setStyleSheet(""" # type: ignore[reportUnknownMemberType]
            QLabel[objectName="videoLabel"] {
                 border: 1px solid red; background-color: #500000; color: white;
                 font-size: 14px; padding: 10px;
            }
        """)

        # Ensure the video thread is stopped if an error occurred (unless it already finished).
        # Check self.video_thread is not None before accessing methods. Keep check, ignore diagnostic.
        # Diagnostic reportUnknownMemberType on isRunning stub addressed by ignore
        if self.video_thread is not None and self.video_thread.isRunning(): # type: ignore[reportUnnecessaryComparison, reportUnknownMemberType]
            self.video_thread.stop()

        # Restore Connect button state.
        # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
        self.connect_source_button.setText("Connect Source") # type: ignore[reportUnknownMemberType]
        # Diagnostic reportUnknownMemberType on setEnabled stub addressed by ignore
        self.connect_source_button.setEnabled(True) # type: ignore[reportUnknownMemberType]
        self.first_frame_received = False # Reset this flag on error

    def get_stylesheet(self) -> str:
        """Returns the application's custom CSS stylesheet."""
        return """
            QMainWindow, QWidget { background-color: #2e2e2e; color: #e0e0e0; }
            QGroupBox { border: 1px solid #4f4f4f; margin-top: 1ex; padding-top:15px; font-weight: bold;}
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; background-color: #2e2e2e;}
            QPushButton { background-color: #555; color: #fff; border: 1px solid #666; padding: 6px; border-radius: 3px; min-height: 22px; }
            QPushButton:hover { background-color: #6a6a6a; } QPushButton:pressed { background-color: #4a4a4a; }
            QPushButton:disabled { background-color: #444; color: #888; border-color: #555;}
            QLabel { color: #e0e0e0; background-color: transparent; }
            QListWidget { background-color: #3a3a3a; border: 1px solid #4f4f4f; selection-background-color: #0078d7;} /* Add selection style */
            QListWidget::item { border-bottom: 1px solid #4f4f4f; }
            QListWidget::item QLabel { color: #e0e0e0; background-color: transparent; padding: 5px 2px; }
            QListWidget::item:selected { background-color: #0078d7; border-bottom: 1px solid #0053a0; }
            QListWidget::item:selected QLabel { color: #ffffff; }
            QListWidget::item:hover { background-color: #4a4a4a; }
            QCheckBox { spacing: 5px; background-color: transparent; }
            QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #777; border-radius: 3px; }
            QCheckBox::indicator:checked { background-color: #0078d7; border: 1px solid #0053a0; }
            QCheckBox::indicator:unchecked { background-color: #555; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background-color: #3f3f3f; color: #e0e0e0; border: 1px solid #555; padding: 3px; border-radius: 3px; min-height: 20px;}
            QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 16px; }
            QComboBox::down-arrow { image: none; } /* Remove default arrow */
            QComboBox::drop-down { border: none; width: 16px; } /* Remove default drop down border */
            QScrollArea { border: 1px solid #4f4f4f; }
            /* Object names defined for specific QLabel instances */
            QLabel[objectName="videoLabel"], QLabel[objectName="originalVideoLabel"],
            QLabel[objectName="stageInputVideoLabel"], QLabel[objectName="stageOutputVideoLabel"] {
                 border: 1px solid #555; background-color: #222; color: #888;
            }
            /* Specific style for error state on the main video label */
            QLabel[objectName="videoLabel"][style*="border: 1px solid red;"] {
                /* Keep the error style */
            }
        """

    def populate_initial_operations_for_bottles(self) -> None:
        """Adds a predefined set of operations to the pipeline for demonstration."""
        # Define the configuration for the initial pipeline steps.
        # Each entry corresponds to an AvailableOpConfigEntry display_name
        # with an optional user_name, parameter overrides, and enabled state override.
        # Explicitly type the list and its contents
        pipeline_to_add: List[PipelineToAddEntry] = [
            {"config_name": "Undistort", "user_name": "Undistort", "is_enabled_override": False}, # Start disabled
            {"config_name": "Resize", "user_name": "Resize", "params_override": {"width": 640, "height": 480, "interpolation": "area"}, "is_enabled_override": True},
            {"config_name": "Grayscale", "user_name": "Grayscale"},
            {"config_name": "Gaussian Blur", "user_name": "Blur", "params_override": {"ksize_w": 5, "ksize_h": 5}},
            {"config_name": "Canny Edge", "user_name": "Edges", "params_override": {"threshold1": 50.0, "threshold2": 150.0}},
            {"config_name": "Find Contours", "user_name": "Find Shapes", "params_override": {"mode": "external", "min_area_filter": 500.0}},
            {"config_name": "Filter Contours", "user_name": "Filter Bottles", "params_override": {"min_area": 500.0, "max_area": 50000.0, "min_aspect_ratio": 0.2, "max_aspect_ratio": 1.5, "min_circularity":0.1, "max_circularity":0.8}}
        ]

        # Iterate through the list of operations to add
        for item_cfg in pipeline_to_add: # item_cfg is PipelineToAddEntry (a TypedDict)
            # Find the corresponding configuration in available_op_configs by display_name
            config_opt: Optional[AvailableOpConfigEntry] = next(
                (c for c in self.available_op_configs if c["display_name"] == item_cfg["config_name"]), None
            )
            # Check is needed runtime, ignore diagnostic.
            if config_opt: # type: ignore[reportUnnecessaryComparison]
                # Get user-specified name or use display_name
                user_name = item_cfg.get("user_name", config_opt["display_name"])
                # Get enabled state override or default to True
                # Use .get with default None, then check explicitly for None.
                is_enabled_override_val: Optional[bool] = item_cfg.get("is_enabled_override", None)

                # Add the new operation instance. _add_new_operation_from_config returns the created op.
                # We need the returned op to apply parameter overrides. new_op is guaranteed ImageOperation.
                new_op: ImageOperation = self._add_new_operation_from_config(config_opt, user_given_name=user_name, is_enabled_override=is_enabled_override_val)

                # Apply parameter overrides if provided
                params_override_raw: Any = item_cfg.get("params_override")
                # Necessary because source is Any. Keep check, ignore diagnostic.
                # Check if new_op was actually created (should be unless _add_new_operation_from_config failed unexpectedly)
                if isinstance(params_override_raw, dict): # type: ignore[reportUnnecessaryIsInstance] # Keep check on raw dict
                    # Cast the override dict values to ParamValue for type safety during iteration.
                    # Keys are assumed str from the way the literal is defined.
                    params_override_dict: Dict[str, ParamValue] = cast(Dict[str, ParamValue], params_override_raw)

                    for p_name, p_val_override in params_override_dict.items():
                        # Check if the parameter exists in the new operation before overriding
                        if p_name in new_op.params:
                            # Update the 'value' field for the parameter in the operation's params dict.
                            # Ensure p_val_override is compatible with ParamValue union, though not strictly checked here.
                            # The to_int_robust/to_float_robust calls in the processing functions handle runtime type issues.
                            # The ParamConfigValue TypedDict requires 'value' to be ParamValue.
                            # The override value `p_val_override` is assumed to be compatible `ParamValue`.
                            new_op.params[p_name]["value"] = p_val_override
                        else:
                            print(f"Warning: Param '{p_name}' not found for override in operation '{new_op.base_name}' (Type: {new_op.op_type_name}).")
            elif "config_name" in item_cfg: # Check if config_name was present in the item_cfg
                print(f"Warning: Configuration for operation '{item_cfg['config_name']}' not found in available configs. Skipping initial add.")
            # Check is needed runtime, ignore diagnostic.
            else: # type: ignore[reportUnnecessaryComparison]
                 print(f"Warning: Invalid initial operation configuration item missing 'config_name'. Skipping.")


        # After adding all initial operations, update the internal data list from the widget
        self._update_operations_data_from_widget()

        # Select the first item in the list if the list is not empty
        # Diagnostic reportUnknownMemberType on count stub addressed by ignore
        if self.operations_list_widget.count() > 0: # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on setCurrentRow stub addressed by ignore
            self.operations_list_widget.setCurrentRow(0) # type: ignore[reportUnknownMemberType]

    def _add_operation_to_list_widget_item(self, operation_obj: ImageOperation) -> None:
        """Creates a OperationListItem for the given operation and adds it to the list widget."""
        # Create our custom list item object, passing the operation and the parent list widget.
        list_item = OperationListItem(operation_obj, self.operations_list_widget)
        # Connect the item's emitter signal for enabled state changes to the main window's handler.
        # This connection object is stored to allow disconnection later (e.g., on item removal).
        # The signal operation_enabled_changed emits uuid.UUID and bool. The slot accepts uuid.UUID and bool.
        # The types match. Standard connection. Add ignore.
        # Diagnostic reportUnknownMemberType on connect stub addressed by ignore
        self._connections.append(list_item.emitter.operation_enabled_changed.connect(self.handle_operation_enabled_changed_slot)) # type: ignore[reportUnknownMemberType]


    def _add_new_operation_from_config(self, config: AvailableOpConfigEntry,
                                     user_given_name: Optional[str] = None,
                                     is_enabled_override: Optional[bool] = None) -> ImageOperation:
        """Creates a new ImageOperation instance based on a config and adds it to the UI list."""
        # Determine the base name, handling potential duplicates by adding a counter.
        op_display_name_candidate = user_given_name if user_given_name else config["display_name"]
        base_name = op_display_name_candidate
        count = 1
        # Build a list of existing names for duplication check from the internal data list
        existing_names = [op.base_name for op in self.operations_data]
        final_op_display_name = op_display_name_candidate
        while final_op_display_name in existing_names:
            final_op_display_name = f"{base_name} ({count})"
            count += 1

        # Determine the initial enabled state, using override if provided, otherwise default True.
        # Check is needed runtime, ignore diagnostic.
        initial_is_enabled = is_enabled_override if is_enabled_override is not None else True # type: ignore[reportUnnecessaryComparison]

        # Deep copy the default parameters from the config to avoid modifying the global config dict.
        # Ensure ParamConfigValue structure is maintained during copy.
        default_params_copy: Dict[str, ParamConfigValue] = {}
        for k, v_config in config["params"].items(): # v_config is ParamConfigValue
             default_params_copy[k] = v_config.copy() # Shallow copy of the inner dictionary

        # Look up the function reference from the processing_functions map.
        function_ref: Optional[ProcessingFunction] = self.processing_functions.get(config["op_type_name"])
        # Check is needed runtime, ignore diagnostic.
        if function_ref is None: # type: ignore[reportUnnecessaryComparison]
             print(f"Warning: No processing function found for type '{config['op_type_name']}'. Operation '{final_op_display_name}' will be inactive.")

        # Create the new ImageOperation instance.
        # Pass the validated/typed params
        new_op = ImageOperation(base_name=final_op_display_name,
                                op_type_name=config["op_type_name"],
                                function_ref=function_ref,
                                default_params=default_params_copy,
                                is_enabled=initial_is_enabled)

        # Add the newly created operation to the QListWidget as a custom item.
        # Use the dedicated helper method for adding existing operation objects.
        self._add_operation_to_list_widget_item(new_op)

        return new_op # Return the created operation instance (guaranteed non-Optional)


    def prompt_add_operation(self) -> None:
        """Prompts the user to select an operation type to add to the pipeline."""
        # Create a list of display names for the input dialog. Include a short description.
        items_with_desc: List[str] = [] # Type hint as List[str]
        for config_entry in self.available_op_configs: # config_entry is AvailableOpConfigEntry
            item_text = config_entry["display_name"]
            op_type_key = config_entry.get("op_type_name", config_entry["display_name"])

            # Get the detailed explanation for the operation type
            op_detail_entry: Optional[OpExplanationEntry] = self.operation_details.get(op_type_key)
            full_explanation_str: str = ""
            # Check is needed runtime, ignore diagnostic.
            if op_detail_entry: # type: ignore[reportUnnecessaryComparison]
                base_explanation_raw: Any = op_detail_entry.get("explanation", "")
                # Ensure explanation is a string. Necessary check as source is Any.
                # Check is needed runtime, ignore diagnostic.
                if isinstance(base_explanation_raw, str): # type: ignore[reportUnnecessaryIsInstance]
                     full_explanation_str = base_explanation_raw
                # Check is needed runtime, ignore diagnostic.
                elif base_explanation_raw is not None: # type: ignore[reportUnnecessaryComparison]
                     print(f"Warning: Explanation for '{op_type_key}' has invalid format (not a string).")


            # Take the first paragraph of the explanation as a short description
            short_desc_parts = full_explanation_str.split("\n\n")
            # Clean up basic HTML tags and newlines for the short description
            short_desc_cleaned = short_desc_parts[0].replace("\n", " ").replace("<b>","").replace("</b>","") if short_desc_parts else ""

            # Append short description if available, truncating if necessary
            if short_desc_cleaned:
                max_len = 70
                # Concatenate strings
                item_text += f" – {short_desc_cleaned[:max_len]}{'...' if len(short_desc_cleaned) > max_len else ''}"
            items_with_desc.append(item_text)

        # Show the input dialog with the list of available operations.
        # Standard PyQt method. Add ignore.
        # chosen_item_text_opt can be None if the user cancels.
        # Diagnostic reportUnknownMemberType on getItem stub addressed by ignore
        chosen_item_text_opt, ok = QInputDialog.getItem(self, "Add Operation", "Select operation type:", items_with_desc, 0, False) # type: ignore[reportUnknownMemberType]

        # If the user selected an item and clicked OK. The check is necessary runtime logic. Ignore diagnostic.
        if ok and chosen_item_text_opt is not None: # type: ignore[reportUnnecessaryComparison]
            # Extract the display name (before the ' – ' separator)
            # Use split with maxsplit=1 to handle potential ' – ' in the description
            chosen_op_display_name = chosen_item_text_opt.split(" – ", 1)[0]

            # Find the corresponding configuration entry using the extracted display name
            selected_config: Optional[AvailableOpConfigEntry] = next((c for c in self.available_op_configs if c["display_name"] == chosen_op_display_name), None)

            # Check is needed runtime, ignore diagnostic.
            if selected_config: # type: ignore[reportUnnecessaryComparison]
                # Add the new operation based on the selected configuration.
                # We need to capture the returned op to update its parameters later if needed,
                # but for this simple add operation, we don't need the return value here.
                self._add_new_operation_from_config(selected_config)

                # Update the internal operations data list.
                self._update_operations_data_from_widget()
                # Select the newly added item (it will be the last one).
                # Diagnostic reportUnknownMemberType on count stub addressed by ignore
                if self.operations_list_widget.count() > 0: # type: ignore[reportUnknownMemberType]
                     # Diagnostic reportUnknownMemberType on setCurrentRow stub addressed by ignore
                     self.operations_list_widget.setCurrentRow(self.operations_list_widget.count() - 1) # type: ignore[reportUnknownMemberType]
                # Re-process the current frame to show the effect of the new operation.
                self.process_and_display_frame_slot()
            else:
                # This case should ideally not happen if the list was built correctly
                print(f"Error: Selected configuration '{chosen_op_display_name}' not found after dialog.")

    def remove_selected_operation(self) -> None:
        """Removes the currently selected operation from the pipeline."""
        # Get the currently selected list widget item.
        # Diagnostic reportUnknownMemberType on currentItem stub addressed by ignore
        current_list_item: Optional[QListWidgetItem] = self.operations_list_widget.currentItem() # type: ignore[reportUnknownMemberType]

        # Check if an item is selected and if it's our custom OperationListItem.
        # The comparison is necessary runtime logic. Ignore diagnostic.
        if not isinstance(current_list_item, OperationListItem): # type: ignore[reportUnnecessaryComparison]
            # Diagnostic reportUnknownMemberType on information stub addressed by ignore
            QMessageBox.information(self, "Remove", "No operation selected."); return # type: ignore[reportUnknownMemberType]

        # Cast the selected item to our specific type
        current_op_list_item: OperationListItem = current_list_item
        op_name = current_op_list_item.operation.base_name # Get the operation's name

        # Ask for user confirmation before removing.
        # Diagnostic reportUnknownMemberType on question stub and enum access addressed by ignore
        reply = QMessageBox.question(self, "Confirm Remove", f"Remove '{op_name}'?", # type: ignore[reportUnknownMemberType]
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, # type: ignore[reportUnknownMemberType]
                                     QMessageBox.StandardButton.No) # type: ignore[reportUnknownMemberType]

        # Diagnostic reportUnknownMemberType on enum access addressed by ignore
        if reply == QMessageBox.StandardButton.Yes: # type: ignore[reportUnknownMemberType]
            # Get the row index of the item.
            # Diagnostic reportUnknownMemberType on row stub addressed by ignore
            row = self.operations_list_widget.row(current_op_list_item) # type: ignore[reportUnknownMemberType]

            # Disconnect signals emitted *by* this item's emitter before removing the item.
            # The item widget is the parent of the emitter, and the item widget is owned by the QListWidgetItem.
            # When the QListWidgetItem is taken from the list, it (and its children) are deleted.
            # Qt's parent-child mechanism *should* handle disconnection of the emitter's signals.
            # Explicit manual disconnection loop for this item's connections is removed,
            # relying on deleteLater() and the global cleanup in closeEvent.
            pass

            # The OperationListItem and its children (item_widget, emitter) will be deleted by takeItem(row).
            # This triggers Qt's cleanup, including disconnecting signals where it's the sender.
            # We also clean up any *other* connections in closeEvent or full load.

            # Remove the item from the list widget. This deletes the item and its children.
            # The removed_item variable is not needed.
            # Diagnostic reportUnknownMemberType on takeItem stub addressed by ignore
            self.operations_list_widget.takeItem(row) # type: ignore[reportUnknownMemberType]

            # Update the internal operations data list from the remaining items.
            self._update_operations_data_from_widget()
            # Re-process the current frame to reflect the removal.
            self.process_and_display_frame_slot()

            # Select a new item after removal: the one at the same index (if available), or the last one if the removed item was the last.
            new_selection_row = -1
            # Diagnostic reportUnknownMemberType on count stub addressed by ignore
            if self.operations_list_widget.count() > 0: # type: ignore[reportUnknownMemberType]
                # Diagnostic reportUnknownMemberType on count stub addressed by ignore
                new_selection_row = min(row, self.operations_list_widget.count() - 1) # type: ignore[reportUnknownMemberType]

            if new_selection_row >= 0:
                 # Ensure the new selection is a valid row index
                 # Diagnostic reportUnknownMemberType on count stub addressed by ignore
                 if new_selection_row < self.operations_list_widget.count(): # type: ignore[reportUnknownMemberType]
                    # Diagnostic reportUnknownMemberType on setCurrentRow stub addressed by ignore
                    self.operations_list_widget.setCurrentRow(new_selection_row) # type: ignore[reportUnknownMemberType]
            else: # No items left in the list
                self.on_operation_selected(None, None) # Clear parameter panel and stage views


    # New helper slot to receive rowsMoved signal parameters and call the logic.
    def _handle_list_widget_rows_moved(self,
                                       parent_idx: QModelIndex, start: int, end: int,
                                       dest_parent_idx: QModelIndex, dest_row: int) -> None:
         """Handles rowsMoved signal parameters and triggers pipeline update."""
         # Parameters are typed explicitly here, resolving reportUnknownLambdaType.
         _ = parent_idx; _ = start; _ = end; _ = dest_parent_idx; _ = dest_row # Mark as unused
         self.on_pipeline_reordered() # Call the method that contains the update logic

    def on_pipeline_reordered(self) -> None:
        """Slot called when the operations list is reordered by dragging and dropping."""
        print(f"Pipeline reordered. Updating internal data.")
        # Rebuild the internal operations_data list to match the new order.
        # The QListWidget's internal model order has changed, so we just need to
        # resynchronize our list from the widget's new state.
        self._update_operations_data_from_widget()
        # Re-process the current frame to show the pipeline with the new order.
        self.process_and_display_frame_slot()


    def _update_operations_data_from_widget(self) -> None:
        """
        Rebuilds the internal self.operations_data list based on the current order
        and contents of the operations_list_widget. Also updates item numbering in UI.
        """
        self.operations_data.clear() # Clear the existing list
        # Diagnostic reportUnknownMemberType on count stub addressed by ignore
        for i in range(self.operations_list_widget.count()): # type: ignore[reportUnknownMemberType]
            # Get the item at index i. This is a QListWidgetItem.
            # Diagnostic reportUnknownMemberType on item stub addressed by ignore
            item: Optional[QListWidgetItem] = self.operations_list_widget.item(i) # type: ignore[reportUnknownMemberType]
            # We expect it to be an OperationListItem (our custom subclass)
            # The comparison is necessary runtime logic. Ignore diagnostic.
            if isinstance(item, OperationListItem): # type: ignore[reportUnnecessaryComparison]
                item_op_list_item: OperationListItem = item
                # Update the label text to show the correct row number
                item_op_list_item.update_label_text()
                # Append the ImageOperation object from the item to the internal data list
                self.operations_data.append(item_op_list_item.operation)
            # Check is needed runtime, ignore diagnostic.
            else: # type: ignore[reportUnnecessaryComparison]
                 # Handle unexpected item types if necessary
                 print(f"Warning: Item at index {i} is not an OperationListItem.")


    def on_operation_selected(self, current_item_widget: Optional[QListWidgetItem],
                              _previous_item_widget: Optional[QListWidgetItem]) -> None:
        """Slot called when the selected item in the operations list changes."""
        # Parameters emitted by QListWidget.currentItemChanged signal:
        # current (QListWidgetItem | None), previous (QListWidgetItem | None)
        # We ignore the previous item. Lambda parameters were casted in the connect call.
        _ = _previous_item_widget

        # Cast the generic QListWidgetItem to our specific OperationListItem if it's one.
        current_op_list_item: Optional[OperationListItem] = None
        # Check is necessary runtime logic. Ignore diagnostic.
        if isinstance(current_item_widget, OperationListItem): # type: ignore[reportUnnecessaryComparison]
            current_op_list_item = current_item_widget

        # Ensure the layout for parameter widgets and the explanation label exist.
        # These are instance variables initialized in init_ui.
        # Added explicit checks for None before accessing layout/label.
        # Check is needed runtime, ignore diagnostic.
        # Check is needed runtime, ignore diagnostic.
        if self.parameter_widgets_area_layout is None or self.operation_explanation_label is None: # type: ignore[reportUnnecessaryComparison, reportUnnecessaryComparison] # Check is technically unnecessary after init_ui, but safe
             print("Error: Parameter panel layout or explanation label not initialized.")
             return # Should not happen if init_ui ran successfully

        current_layout = self.parameter_widgets_area_layout # Assign to non-optional local variable for cleaner code

        # --- Clear existing parameter widgets ---
        # Remove all widgets/layouts that were added for the previous operation's parameters.
        # These are the items starting after the initial fixed labels (Title, OpNameLabel, ExplanationLabel).
        # Items to remove are from index 3 onwards.

        # First, find and remove the stretch item if it's the last one.
        # It's safer to iterate and check type rather than assuming index.
        # The layout contains QWidgetItem or QSpacerItem.
        # Diagnostic reportUnknownMemberType on count stub addressed by ignore
        last_idx = current_layout.count() - 1 # type: ignore[reportUnknownMemberType]
        if last_idx >= 0:
            # Diagnostic reportUnknownMemberType on itemAt stub addressed by ignore
            last_item = current_layout.itemAt(last_idx) # type: ignore[reportUnknownMemberType]
            # Check if the item is a spacer using its `spacerItem()` method
            # Check is needed runtime, ignore diagnostic.
            # Diagnostic reportUnknownMemberType on spacerItem stub addressed by ignore
            if last_item is not None and last_item.spacerItem() is not None: # type: ignore[reportUnnecessaryComparison, reportUnknownMemberType]
                # Diagnostic reportUnknownMemberType on takeAt stub addressed by ignore
                current_layout.takeAt(last_idx) # Removes the spacer item # type: ignore[reportUnknownMemberType]

        # Now, remove all items (widgets or layouts) that are not the fixed header labels.
        # Loop and remove items starting from index 3.
        # Removing items shifts subsequent items, so always remove from the same index (index 3).
        # Diagnostic reportUnknownMemberType on count stub addressed by ignore
        while current_layout.count() > 3: # type: ignore[reportUnknownMemberType]
            # Take the item at index 3 (which is the first parameter widget/layout container)
            # Diagnostic reportUnknownMemberType on takeAt stub addressed by ignore
            item_to_remove = current_layout.takeAt(3) # type: ignore[reportUnknownMemberType]
            # Check is needed runtime, ignore diagnostic.
            if item_to_remove: # type: ignore[reportUnnecessaryComparison]
                # Diagnostic reportUnknownMemberType on widget stub addressed by ignore
                widget = item_to_remove.widget() # Get the widget if the item is a QWidgetItem # type: ignore[reportUnknownMemberType]
                # Check is needed runtime, ignore diagnostic.
                if widget: # type: ignore[reportUnnecessaryComparison]
                    # Diagnostic reportUnknownMemberType on setParent stub addressed by ignore
                    widget.setParent(None) # Unparent the widget from the layout # type: ignore[reportUnknownMemberType]
                    # The widget and its children (like control widgets inside parameter frames)
                    # will be marked for deletion by deleteLater().
                    # Diagnostic reportUnknownMemberType on deleteLater stub addressed by ignore
                    widget.deleteLater() # type: ignore[reportUnknownMemberType]
                else:
                    # If it's not a widget item, it must be a layout item containing a layout
                    # Diagnostic reportUnknownMemberType on layout stub addressed by ignore
                    nested_layout_item = item_to_remove.layout() # type: ignore[reportUnknownMemberType]
                    # Check is needed runtime, ignore diagnostic.
                    if nested_layout_item: # type: ignore[reportUnnecessaryComparison]
                        # Recursively remove and delete widgets/layouts within the nested layout
                        # While loop and takeAt(0) is the standard way to clear a layout
                        # Diagnostic reportUnknownMemberType on count stub addressed by ignore
                        while nested_layout_item.count(): # type: ignore[reportUnknownMemberType]
                            # Diagnostic reportUnknownMemberType on takeAt stub addressed by ignore
                            child_item = nested_layout_item.takeAt(0) # type: ignore[reportUnknownMemberType]
                            # Check is needed runtime, ignore diagnostic.
                            if child_item: # type: ignore[reportUnnecessaryComparison]
                                # Diagnostic reportUnknownMemberType on widget stub addressed by ignore
                                child_widget = child_item.widget() # type: ignore[reportUnknownMemberType]
                                # Check is needed runtime, ignore diagnostic.
                                if child_widget: # type: ignore[reportUnnecessaryComparison]
                                    # Diagnostic reportUnknownMemberType on setParent stub addressed by ignore
                                    child_widget.setParent(None) # type: ignore[reportUnknownMemberType]
                                    # Diagnostic reportUnknownMemberType on deleteLater stub addressed by ignore
                                    child_widget.deleteLater() # type: ignore[reportUnknownMemberType]
                                # If there are nested layouts within this one, they would also need handling,
                                # but our parameter structure is flat (widget inside QHBoxLayout inside QFrame).
                                # If it's a nested layout item (QLayoutItem but not QWidgetItem), its layout() is non-None
                                # Diagnostic reportUnknownMemberType on layout stub addressed by ignore
                                nested_child_layout = child_item.layout() # type: ignore[reportUnknownMemberType]
                                # Check is needed runtime, ignore diagnostic.
                                if nested_child_layout: # type: ignore[reportUnnecessaryComparison]
                                    # In this app's structure, this path is unlikely unless parameter controls become complex layouts.
                                    print("Warning: Found unexpected nested layout during parameter panel cleanup.")
                                    # Manual deletion of the nested layout object might be needed here, e.g., nested_child_layout.deleteLater()
                                    # Diagnostic reportUnknownMemberType on deleteLater stub addressed by ignore
                                    nested_child_layout.deleteLater() # type: ignore[reportUnknownMemberType]
                                    pass # Continue loop after handling child widget or layout

                        # Delete the nested layout object itself.
                        # Diagnostic reportUnknownMemberType on deleteLater stub addressed by ignore
                        nested_layout_item.deleteLater() # type: ignore[reportUnknownMemberType]


        # --- Populate parameter widgets for the newly selected operation ---
        # Check is needed runtime, ignore diagnostic.
        if current_op_list_item: # type: ignore[reportUnnecessaryComparison]
            op = current_op_list_item.operation # Get the associated ImageOperation object

            # Update operation name label
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            self.parameter_panel_op_name_label.setText(f"<b>{op.base_name}</b> (Type: {op.op_type_name})") # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on setStyleSheet stub addressed by ignore
            self.parameter_panel_op_name_label.setStyleSheet("font-style: normal; color: #e0e0e0; margin-bottom: 5px; font-size: 10pt;") # type: ignore[reportUnknownMemberType]

            # Update operation explanation label
            op_details_for_type: Optional[OpExplanationEntry] = self.operation_details.get(op.op_type_name)
            explanation_html_parts: List[str] = [] # List of HTML strings
            base_explanation: str = "No detailed explanation available."
            param_explanations_dict: Optional[Dict[str,str]] = None

            # Check is needed runtime, ignore diagnostic.
            if op_details_for_type: # type: ignore[reportUnnecessaryComparison]
                base_explanation_raw: Any = op_details_for_type.get("explanation", base_explanation)
                # Ensure explanation is a string. Necessary check as source is Any.
                # Check is needed runtime, ignore diagnostic.
                if isinstance(base_explanation_raw, str): # type: ignore[reportUnnecessaryIsInstance]
                     base_explanation = base_explanation_raw
                # Check is needed runtime, ignore diagnostic.
                elif base_explanation_raw is not None: # type: ignore[reportUnnecessaryComparison]
                     print(f"Warning: Explanation for '{op.op_type_name}' has invalid format (not a string).")


                raw_param_explanations: Any = op_details_for_type.get("param_explanations")
                # Validate that param_explanations is a dict. Necessary check as source is Any. Ignore diagnostic.
                # Check is needed runtime, ignore diagnostic.
                if isinstance(raw_param_explanations, dict): # type: ignore[reportUnnecessaryIsInstance]
                    # Ensure keys/values are strings for the dictionary
                    # This cast is okay as we're converting keys/values to str immediately.
                    param_explanations_dict = {str(k): str(v) for k, v in cast(Dict[Any, Any], raw_param_explanations).items()}
                # Check is needed runtime, ignore diagnostic.
                elif raw_param_explanations is not None: # type: ignore[reportUnnecessaryComparison]
                     print(f"Warning: Explanation data for '{op.op_type_name}' has invalid 'param_explanations' format (not a dict).")


            # Add base explanation HTML
            # Replace double newlines with <br><br> for paragraphs, single newlines with <br>
            explanation_html_parts.append("<p>" + base_explanation.replace("\n\n", "<br><br>").replace("\n", "<br>") + "</p>")

            # Add parameter details HTML if parameters exist and explanations are available
            # Check is needed runtime, ignore diagnostic.
            if param_explanations_dict and op.params: # type: ignore[reportUnnecessaryComparison]
                explanation_html_parts.append("<br><b>Parameter Details:</b><ul style='margin-left: -20px; list-style-type: disc;'>")
                # Iterate through the operation's actual parameters
                for param_key_in_op_params in op.params.keys():
                    # Format parameter name for display (capitalize and replace underscores)
                    param_display_name = param_key_in_op_params.replace('_', ' ').title()
                    # Get parameter explanation, default if not found
                    # Use get() with a default empty string to handle missing keys in param_explanations_dict
                    param_explanation_from_details = str(param_explanations_dict.get(param_key_in_op_params, ""))
                    # Remove HTML tags and convert <br> to newlines for tooltip
                    final_tooltip = param_explanation_from_details.replace("<b>","").replace("</b>","").replace("<br>","\n")
                    # Convert newlines in explanation to HTML breaks for the parameter list
                    param_desc_html_br = param_explanation_from_details.replace("\n", "<br>") # Use the original explanation for HTML

                    # Append list item HTML
                    explanation_html_parts.append(f"<li style='margin-bottom: 4px;'><b>{param_display_name}</b>: {param_desc_html_br}</li>")
                explanation_html_parts.append("</ul>")

            # Join all HTML parts and set the label text
            # Check if operation_explanation_label is not None before setting text.
            # It's non-Optional after init_ui, but safe check. Ignore diagnostic.
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            if self.operation_explanation_label: # type: ignore[reportUnnecessaryComparison]
                 self.operation_explanation_label.setText("".join(explanation_html_parts)) # type: ignore[reportUnknownMemberType]


            # --- Create and add parameter control widgets ---
            for param_name, param_props in op.params.items(): # param_props is ParamConfigValue
                # Create a frame for each parameter for visual separation
                param_container = QFrame(); param_container.setFrameShape(QFrame.Shape.StyledPanel) # setFrameShape stub
                # Diagnostic reportUnknownMemberType on setStyleSheet stub addressed by ignore
                param_container.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 3px; margin-bottom: 5px; padding: 3px; }") # type: ignore[reportUnknownMemberType]
                param_layout = QHBoxLayout(param_container); param_layout.setContentsMargins(3,3,3,3) # setContentsMargins stub

                # Add parameter name label
                label_text = param_name.replace('_', ' ').title()
                label = QLabel(f"{label_text}:"); label.setFixedWidth(120) # setFixedWidth stub
                param_layout.addWidget(label)

                control_widget: Optional[QWidget] = None # Declare control widget variable
                param_type = param_props["type"] # Get parameter type
                current_val = param_props["value"] # Get current value

                # Capture variables for use in lambda functions (values needed inside lambda scope)
                op_id_captured: uuid.UUID = op.id # Capture UUID
                param_name_captured: str = param_name # Capture parameter name

                # Create the appropriate widget based on parameter type
                if param_type == "int":
                    spinbox = QSpinBox()
                    # Set range and single step, providing defaults
                    spinbox.setRange(to_int_robust(param_props.get("min"), -2147483648), to_int_robust(param_props.get("max"), 2147483647)) # setRange stub
                    spinbox.setValue(to_int_robust(current_val)) # setValue stub
                    spinbox.setSingleStep(to_int_robust(param_props.get("step"), 1)) # setSingleStep stub
                    # Connect valueChanged signal (emits int) to update slot using lambda
                    # The lambda receives an int value directly. Explicitly hint param type. Standard connection. Add ignore.
                    # Diagnostic reportUnknownMemberType on valueChanged.connect stub addressed by ignore
                    # Diagnostic reportUnknownLambdaType on lambda parameters addressed by ignore
                    slot_callable_int: Callable[[int], None] = lambda value: self.update_op_param_slot(op_id_captured, param_name_captured, "int", cast(int, value)) # type: ignore[reportUnknownLambdaType]
                    self._connections.append(spinbox.valueChanged.connect(slot_callable_int)) # type: ignore[reportUnknownMemberType]
                    control_widget = spinbox
                elif param_type == "float":
                    dsbox = QDoubleSpinBox()
                    # Set range, value, step, and decimals, providing defaults
                    dsbox.setRange(to_float_robust(param_props.get("min"), -1.0e18), to_float_robust(param_props.get("max"), 1.0e18)) # setRange stub
                    dsbox.setValue(to_float_robust(current_val)) # setValue stub
                    dsbox.setSingleStep(to_float_robust(param_props.get("step"), 0.1)) # setSingleStep stub
                    dsbox.setDecimals(to_int_robust(param_props.get("decimals"), 2)) # setDecimals stub
                    # Connect valueChanged signal (emits float) to update slot using lambda
                    # The lambda receives a float value directly. Explicitly hint param type. Standard connection. Add ignore.
                    # Diagnostic reportUnknownMemberType on valueChanged.connect stub addressed by ignore
                    # Diagnostic reportUnknownLambdaType on lambda parameters addressed by ignore
                    slot_callable_float: Callable[[float], None] = lambda value: self.update_op_param_slot(op_id_captured, param_name_captured, "float", cast(float, value)) # type: ignore[reportUnknownLambdaType]
                    self._connections.append(dsbox.valueChanged.connect(slot_callable_float)) # type: ignore[reportUnknownMemberType]
                    control_widget = dsbox
                elif param_type == "choice":
                    combo = QComboBox()
                    # Get choices dictionary, ensure it's a dict. Default to empty dict if not.
                    choices_dict_raw: Any = param_props.get("choices", {})
                    choices_dict: Dict[str, ParamValue] = {}
                    # Necessary because choices_dict_raw comes from Any. Keep check, ignore diagnostic.
                    # Check is needed runtime, ignore diagnostic.
                    if isinstance(choices_dict_raw, dict): # type: ignore[reportUnnecessaryIsInstance]
                         choices_dict = cast(Dict[str, ParamValue], choices_dict_raw)

                    for disp_txt, act_val in choices_dict.items():
                        # addItem accepts str, userData can be Any (will be stored as QVariant internally)
                        # Ensure disp_txt is string
                        # Diagnostic reportUnknownMemberType on addItem stub addressed by ignore
                        combo.addItem(str(disp_txt), userData=act_val) # type: ignore[reportUnknownMemberType]

                    # Find the index matching the current value and set it
                    # findData returns int.
                    # Diagnostic reportUnknownMemberType on findData stub addressed by ignore
                    idx: int = combo.findData(current_val) # type: ignore[reportUnknownMemberType]
                    # Set current index, handle -1 (not found) or empty combo
                    if idx != -1:
                         # Diagnostic reportUnknownMemberType on setCurrentIndex stub addressed by ignore
                         combo.setCurrentIndex(idx) # type: ignore[reportUnknownMemberType]
                    # Diagnostic reportUnknownMemberType on count stub addressed by ignore
                    elif combo.count() > 0: # type: ignore[reportUnknownMemberType] # type: ignore[reportUnknownMemberType] # count stub might be generic
                         # Diagnostic reportUnknownMemberType on setCurrentIndex stub addressed by ignore
                         combo.setCurrentIndex(0) # type: ignore[reportUnknownMemberType] # Default to first item if current value not found
                    else:
                         # Combo is empty, disable it or handle appropriately
                         # Diagnostic reportUnknownMemberType on setEnabled stub addressed by ignore
                         combo.setEnabled(False) # type: ignore[reportUnknownMemberType]
                         print(f"Warning: Choices combo for param '{param_name}' is empty.")

                    # Connect currentIndexChanged signal (emits int) to update slot using lambda
                    # The lambda parameter 'index' is not strictly needed for the slot,
                    # as the slot will get the current data from the combobox reference. Standard connection. Add ignore.
                    combo_captured = combo # Capture the combo widget reference
                    # index parameter is emitted by the signal, but the slot implementation doesn't use it directly.
                    # It retrieves the data from the captured combobox reference.
                    # Diagnostic reportUnknownMemberType on currentIndexChanged.connect stub addressed by ignore
                    # Diagnostic reportUnknownLambdaType on lambda parameters addressed by ignore
                    slot_callable_choice: Callable[[int], None] = lambda index: self.update_op_param_slot_combobox(op_id_captured, param_name_captured, combo_captured) # type: ignore[reportUnknownLambdaType]
                    self._connections.append(combo.currentIndexChanged.connect(slot_callable_choice)) # type: ignore[reportUnknownMemberType]
                    control_widget = combo
                elif param_type == "bool":
                    chk = QCheckBox()
                    # Use bool() for robust conversion to boolean state
                    chk.setChecked(bool(current_val)) # setChecked stub
                    # Connect stateChanged signal (emits int) to update slot using lambda
                    # The lambda receives an int state value (0 or 2). Explicitly hint param type. Standard connection. Add ignore.
                    # Diagnostic reportUnknownMemberType on stateChanged.connect stub addressed by ignore
                    # Diagnostic reportUnknownLambdaType on lambda parameters addressed by ignore
                    slot_callable_bool: Callable[[int], None] = lambda state: self.update_op_param_slot_checkbox(op_id_captured, param_name_captured, cast(int, state)) # type: ignore[reportUnknownLambdaType]
                    self._connections.append(chk.stateChanged.connect(slot_callable_bool)) # type: ignore[reportUnknownMemberType]
                    control_widget = chk
                elif param_type == "text":
                    # Ensure text is set from current_val as string
                    ledit = QLineEdit(str(current_val))
                    # Connect editingFinished signal (emits no args) to update slot using lambda. Standard connection. Add ignore.
                    ledit_captured = ledit # Capture line edit reference
                    # Diagnostic reportUnknownMemberType on editingFinished.connect stub addressed by ignore
                    # Diagnostic reportUnknownLambdaType on lambda parameters addressed by ignore
                    slot_callable_text: Callable[[], None] = lambda: self.update_op_param_slot_text(op_id_captured, param_name_captured, ledit_captured) # type: ignore[reportUnknownLambdaType]
                    self._connections.append(ledit.editingFinished.connect(slot_callable_text)) # type: ignore[reportUnknownMemberType]
                    control_widget = ledit

                # If a control widget was created, add it to the layout and set tooltip.
                # Check is needed runtime, ignore diagnostic.
                if control_widget: # type: ignore[reportUnnecessaryComparison]
                    # Get tooltip text from config or parameter explanations
                    tooltip_text_prop = str(param_props.get("tooltip",""))
                    final_tooltip = tooltip_text_prop
                    # If no tooltip in config, use the parameter explanation text from operation_details
                    # Check is needed runtime, ignore diagnostic.
                    if not final_tooltip and param_explanations_dict: # type: ignore[reportUnnecessaryComparison]
                        # Get explanation, remove HTML tags and convert <br> to newlines for tooltip
                        # Use get() with a default empty string to handle missing keys
                        param_explanation_from_details = str(param_explanations_dict.get(param_name, ""))
                        final_tooltip = param_explanation_from_details.replace("<b>","").replace("</b>","").replace("<br>","\n")

                    if final_tooltip:
                         # Diagnostic reportUnknownMemberType on setToolTip stub addressed by ignore
                         control_widget.setToolTip(final_tooltip) # type: ignore[reportUnknownMemberType]

                    # Add the control widget to the parameter's layout (QHBoxLayout)
                    param_layout.addWidget(control_widget)
                    # Add the parameter's container frame to the main parameter layout (QVBoxLayout)
                    current_layout.addWidget(param_container)

        else: # If no operation is selected (current_item_widget is None)
            # Update labels to indicate no selection
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            self.parameter_panel_op_name_label.setText("<i>No operation selected</i>") # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on setStyleSheet stub addressed by ignore
            self.parameter_panel_op_name_label.setStyleSheet("font-style: italic; color: #aaa; margin-bottom: 10px; font-size: 11pt;") # type: ignore[reportUnknownMemberType]
            # Check operation_explanation_label is not None before accessing. It's non-Optional after init_ui. Ignore diagnostic.
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            if self.operation_explanation_label: # type: ignore[reportUnnecessaryComparison]
                self.operation_explanation_label.setText("<i>Select an operation from the pipeline to see its details and parameters.</i>") # type: ignore[reportUnknownMemberType]


        # Add a stretch item at the end to push parameter widgets to the top
        # Ensure there's only one stretch item at the end by removing any existing one before adding.
        # The cleanup loop above already removed the trailing stretch if it existed.
        # Diagnostic reportUnknownMemberType on addStretch stub addressed by ignore
        current_layout.addStretch(1) # type: ignore[reportUnknownMemberType]


    # --- Parameter Update Slots ---
    # These slots are connected to the parameter control widgets' signals.

    def update_op_param_slot_checkbox(self, operation_id: uuid.UUID, param_name: str, state: int) -> None:
        """Slot to update a boolean parameter from a checkbox state."""
        _ = param_name # Mark as intentionally unused
        # Convert checkbox state integer (0 or 2) to boolean (False or True).
        new_value_bool = (Qt.CheckState(state) == Qt.CheckState.Checked)
        # Call the generic update slot with the boolean value.
        self.update_op_param_slot(operation_id, param_name, "bool", new_value_bool)

    def update_op_param_slot_text(self, operation_id: uuid.UUID, param_name: str, line_edit_ref: QLineEdit) -> None:
        """Slot to update a text parameter from a line edit."""
        _ = param_name # Mark as intentionally unused
        # Get the current text from the LineEdit widget reference.
        # Diagnostic reportUnknownMemberType on text stub addressed by ignore
        new_value_text = line_edit_ref.text() # type: ignore[reportUnknownMemberType]
        # Call the generic update slot with the string value.
        self.update_op_param_slot(operation_id, param_name, "text", new_value_text)

    # Updated slot to receive QComboBox reference directly
    def update_op_param_slot_combobox(self, operation_id: uuid.UUID, param_name: str, combobox_ref: QComboBox) -> None:
        """Slot to update a choice parameter from a combobox selection."""
        _ = param_name # Mark as intentionally unused
        # Get the user data associated with the selected item in the combobox.
        # This data is the actual ParamValue (str, int, float, or bool) stored in the choices dict.
        # currentData() returns Any, so we need to validate its type.
        # Diagnostic reportUnknownMemberType on currentData stub addressed by ignore
        new_choice_value_raw: Any = combobox_ref.currentData() # type: ignore[reportUnknownMemberType]

        # Validate the type before casting to ParamValue
        # ParamValue is Union[str, int, float, bool]. Allow None as it's possible if no item is selected.
        # Check is needed runtime, ignore diagnostic.
        if not isinstance(new_choice_value_raw, (str, int, float, bool, type(None))): # type: ignore[reportUnnecessaryIsInstance]
             print(f"Warning: Combobox selected value for param '{param_name}' has unexpected type: {type(new_choice_value_raw)}. Value: {new_choice_value_raw}. Skipping update.")
             return # Exit if type is unexpected

        # If not None, it must be one of the types in ParamValue based on the check.
        # Check is needed runtime, ignore diagnostic.
        if new_choice_value_raw is None: # type: ignore[reportUnnecessaryComparison]
             print(f"Warning: Combobox selected value for param '{param_name}' is None. Skipping update.")
             return

        # The value new_choice_value_raw IS validated to be one of the ParamValue types.
        # The cast is not strictly necessary IF Pyright's inference path recognizes the isinstance checks refine the type.
        # Removed the cast as flagged previously. Assign directly.
        new_choice_value: ParamValue = new_choice_value_raw

        # Call the generic update slot with the selected value.
        # The type string "choice" is passed for logging/debugging context if needed.
        self.update_op_param_slot(operation_id, param_name, "choice", new_choice_value)


    def update_op_param_slot(self, operation_id: uuid.UUID, param_name: str,
                             expected_type_str: Literal["int", "float", "bool", "text", "choice"], # Literal for known types
                             new_value_raw: Any) -> None:
        """Generic slot to update an operation's parameter value."""
        # Find the operation by its ID.
        op: Optional[ImageOperation] = next((o for o in self.operations_data if o.id == operation_id), None)

        # If the operation and parameter exist
        # The comparison between ImageOperation and None has no overlap. Keep check, ignore diagnostic.
        if op is not None and param_name in op.params: # type: ignore[reportUnnecessaryComparison]
            param_config = op.params[param_name] # Get the parameter configuration (ParamConfigValue TypedDict)

            # Attempt to convert the raw new value from the widget signal to the expected type.
            # This conversion logic mostly duplicates the robust converters, but applied based on expected_type_str.
            # It's important to ensure the type matches the ParamValue union.
            new_value_typed: ParamValue # Declare variable type

            try:
                # Use the robust converters for actual type conversion
                if expected_type_str == "int":
                    new_value_typed = to_int_robust(new_value_raw)
                elif expected_type_str == "float":
                    new_value_typed = to_float_robust(new_value_raw)
                elif expected_type_str == "bool":
                    # Simple bool conversion is usually sufficient for check states (int 0/2)
                    new_value_typed = bool(new_value_raw)
                elif expected_type_str == "text":
                    # String conversion is always possible
                    new_value_typed = str(new_value_raw)
                elif expected_type_str == "choice":
                     # The value from update_op_param_slot_combobox should already be the final ParamValue
                     # We already validated its type in update_op_param_slot_combobox.
                     # Just assign the raw value directly, assuming it's already ParamValue.
                     # Add a final check to ensure it's a valid ParamValue type before assignment.
                     # Check is needed runtime, ignore diagnostic.
                     if isinstance(new_value_raw, (str, int, float, bool)): # type: ignore[reportUnnecessaryIsInstance]
                         new_value_typed = new_value_raw
                     else:
                         print(f"Warning: Final choice value for param '{param_name}' has unexpected type {type(new_value_raw)}. Skipping update.")
                         return # Exit if choice value is invalid type
                else:
                    # This branch should not be reached due to Literal type hint, but defensive
                    print(f"Error: Unhandled parameter type '{expected_type_str}' for param '{param_name}'.")
                    return # Exit if type is unhandled

                # Check if the value has actually changed before updating and processing.
                # This prevents unnecessary re-processing if the widget value didn't change meaningfully.
                if param_config["value"] != new_value_typed:
                    # Update the parameter value in the operation object.
                    param_config["value"] = new_value_typed
                    # Re-process and display the current frame to reflect the parameter change.
                    self.process_and_display_frame_slot()
                    # print(f"Updated param '{param_name}' for op '{op.base_name}' to {new_value_typed}") # Optional: log update

            except (ValueError, TypeError, Exception) as e:
                # Catch errors during type conversion or other unexpected issues
                print(f"Error processing new value '{new_value_raw}' for param '{param_name}' (expected {expected_type_str}): {e}")
                # Optionally, revert the UI widget to the old value or show an error message in the UI.
        else:
            # Handle cases where the operation or parameter is not found (shouldn't happen with correct logic)
            print(f"Warning: Param '{param_name}' or operation ID '{operation_id}' not found during update.")


    def handle_operation_enabled_changed_slot(self, operation_id: uuid.UUID, is_enabled: bool) -> None:
        """Slot handling signal from OperationListItem when enabled state changes."""
        # Find the operation by its ID.
        op: Optional[ImageOperation] = next((o for o in self.operations_data if o.id == operation_id), None)
        # If found and state has changed, update its enabled state and re-process the frame.
        # The comparison between ImageOperation and None has no overlap. Keep check, ignore diagnostic.
        if op is not None and op.is_enabled != is_enabled: # type: ignore[reportUnnecessaryComparison]
            op.is_enabled = is_enabled
            print(f"Operation '{op.base_name}' enabled state changed to {is_enabled}") # Log state change
            self.process_and_display_frame_slot()

    def update_video_frame_slot(self, cv_frame_as_object: object) -> None:
        """
        Slot called when a new frame is received from the VideoThread.
        The frame is received as an object due to PyQt signal limitations with NDArray types.
        """
        # Runtime check to ensure the received object is a NumPy array.
        # The signal is emitted with an NDArrayUInt8, but PyQt signals pass objects.
        # Check is needed runtime, ignore diagnostic.
        if not isinstance(cv_frame_as_object, np.ndarray): # type: ignore[reportUnnecessaryIsInstance]
             print(f"Error: Received frame is not a NumPy array. Type: {type(cv_frame_as_object)}. Skipping frame processing.")
             # Optionally display an error on screen via the error slot
             # Diagnostic reportUnknownMemberType on emit stub addressed by ignore
             self.show_video_error_slot(f"Received unexpected data type from source: {type(cv_frame_as_object)}") # type: ignore[reportUnknownMemberType]
             return

        # The object is guaranteed to be an np.ndarray at this point. Cast it to a generic ndarray.
        # Further refine to NDArrayUInt8 after robust conversion.
        cv_frame_ndarray: np.ndarray[Any, Any] = cast(np.ndarray[Any, Any], cv_frame_as_object)

        # Convert the received frame (np.ndarray, potentially various dtypes) to NDArrayUInt8 consistently.
        # Use the robust conversion helper. This handles dtype conversion and contiguity.
        try:
            frame_to_process: NDArrayUInt8 = _ensure_uint8_image(cv_frame_ndarray)
        except Exception as e:
             print(f"Error converting received frame to uint8: {e}. Skipping frame processing.")
             # Diagnostic reportUnknownMemberType on emit stub addressed by ignore
             self.show_video_error_slot(f"Error converting received frame to uint8: {e}") # type: ignore[reportUnknownMemberType]
             return

        # Update the first_frame_received flag and Connect button text/state on receiving the very first frame.
        if not self.first_frame_received:
            self.first_frame_received = True
            print("First frame received. Source connected.") # Log connection success
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            self.connect_source_button.setText("Stop Stream") # type: ignore[reportUnknownMemberType]
            # Re-enable button after successful connection
            # Diagnostic reportUnknownMemberType on setEnabled stub addressed by ignore
            self.connect_source_button.setEnabled(True) # type: ignore[reportUnknownMemberType]
            # Reset video display stylesheet if it was showing an error
            # Check if video_display_label is initialized before accessing
            # It's non-Optional after init_ui, but safe check. Ignore diagnostic.
            # Diagnostic reportUnknownMemberType on styleSheet/setStyleSheet stub addressed by ignore
            if self.video_display_label and "border: 1px solid red;" in self.video_display_label.styleSheet(): # type: ignore[reportUnnecessaryComparison, reportUnknownMemberType]
                 self.video_display_label.setStyleSheet(self.get_stylesheet()) # type: ignore[reportUnknownMemberType]


        # Store the current frame for processing and displaying.
        # Use a copy to avoid issues if the thread modifies the array after emitting (less common but safe).
        # self.current_cv_frame is Optional[NDArrayUInt8]. NDArrayUInt8 is assigned.
        self.current_cv_frame = frame_to_process.copy()
        # Store a copy of the original frame (uint8) specifically for the 'Original Input' display.
        # Ensure this is a copy of the uint8 version obtained from _ensure_uint8_image
        # self.original_frame_for_display is CaptureFrameType (NDArrayUInt8)
        self.original_frame_for_display = frame_to_process.copy()

        # Trigger the pipeline processing and display updates with the new frame.
        self.process_and_display_frame_slot()


    def _convert_cv_to_pixmap(self, cv_img: Optional[CvMatLike], target_label: QLabel) -> Optional[QPixmap]:
        """
        Converts a CvMatLike (or None) image to a QPixmap, suitable for displaying in a QLabel.
        Scales the pixmap to fit the target label's size while maintaining aspect ratio.
        Returns None if conversion fails or image is empty.
        """
        # Handle empty or None input image
        # Check cv_img is None. Keep check, ignore diagnostic.
        if cv_img is None: # type: ignore[reportUnnecessaryComparison]
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            target_label.setText("Input Image N/A"); # type: ignore[reportUnknownMemberType]
            # Also clear any previous pixmap if the input is None
            # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
            target_label.setPixmap(QPixmap()); return None # type: ignore[reportUnknownMemberType]

        # Convert CvMatLike to np.ndarray for robust handling
        # This handles np.ndarray, cv2.Mat, cv2.UMat -> np.ndarray
        img_ndarray: np.ndarray[Any, Any] = np.asarray(cv_img)

        # Check for empty array after conversion
        if img_ndarray.size == 0:
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            target_label.setText("Empty Image"); # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
            target_label.setPixmap(QPixmap()); return None # type: ignore[reportUnknownMemberType] # Clear previous pixmap

        # Ensure the array is C-contiguous for QImage direct data access, and convert to uint8.
        # _ensure_uint8_image also handles scaling/normalization and returns NDArrayUInt8.
        try:
            frame_uint8: NDArrayUInt8 = _ensure_uint8_image(img_ndarray)
        except Exception as e:
            print(f"Error ensuring uint8 in _convert_cv_to_pixmap: {e}");
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            target_label.setText("Uint8 Conv. Error"); # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
            target_label.setPixmap(QPixmap()); return None # type: ignore[reportUnknownMemberType] # Clear previous pixmap
        # Check for empty image after uint8 conversion
        # Ensure shape has at least 2 dimensions before accessing indices 0 and 1
        if len(frame_uint8.shape) < 2:
             # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
             target_label.setText("Unsupported shape after uint8 conversion"); # type: ignore[reportUnknownMemberType]
             # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
             target_label.setPixmap(QPixmap()); return None # type: ignore[reportUnknownMemberType] # setPixmap stub might be generic

        h, w = frame_uint8.shape[:2]
        if h == 0 or w == 0:
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            target_label.setText("Empty Image"); # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
            target_label.setPixmap(QPixmap()); return None # type: ignore[reportUnknownMemberType] # Clear previous pixmap

        qt_image: Optional[QImage] = None # Declare QImage variable

        try:
            bytes_per_line: int
            if len(frame_uint8.shape) == 3: # Color image (BGR or BGRA)
                ch = frame_uint8.shape[2]
                bytes_per_line = ch * w
                # Check channel count and create QImage with appropriate format
                if ch == 3:
                    # Use Format_BGR888 for OpenCV BGR uint8 images
                    # Pass the data buffer from the contiguous numpy array
                    # Diagnostic reportUnknownMemberType on QImage constructor stub addressed by ignore
                    qt_image = QImage(frame_uint8.data, w, h, bytes_per_line, QImage.Format.Format_BGR888) # type: ignore[reportUnknownMemberType]
                elif ch == 4:
                    # Use Format_ARGB32 for BGRA uint8 images (assuming byte order matches)
                    # Pass the data buffer from the contiguous numpy array
                     # Diagnostic reportUnknownMemberType on QImage constructor stub addressed by ignore
                     qt_image = QImage(frame_uint8.data, w, h, bytes_per_line, QImage.Format.Format_ARGB32) # type: ignore[reportUnknownMemberType]
                else:
                    # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
                    target_label.setText(f"Unsupported 3-dim image with {ch} channels"); # type: ignore[reportUnknownMemberType]
                    # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
                    target_label.setPixmap(QPixmap()); return None # type: ignore[reportUnknownMemberType] # Clear previous pixmap
            elif len(frame_uint8.shape) == 2: # Grayscale image
                bytes_per_line = w # Bytes per line is just width for 8-bit grayscale
                # Use Format_Grayscale8 for uint8 single-channel images
                # Pass the data buffer from the contiguous numpy array
                # Diagnostic reportUnknownMemberType on QImage constructor stub addressed by ignore
                qt_image = QImage(frame_uint8.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8) # type: ignore[reportUnknownMemberType]
            else:
                # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
                target_label.setText(f"Unsupported image shape: {frame_uint8.shape}"); # type: ignore[reportUnknownMemberType]
                # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
                target_label.setPixmap(QPixmap()); return None # type: ignore[reportUnknownMemberType] # Clear previous pixmap

        except Exception as e:
            print(f"Error in _convert_cv_to_pixmap during QImage creation: {e}");
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            target_label.setText("QImage Conv. Error"); # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
            target_label.setPixmap(QPixmap()); return None # type: ignore[reportUnknownMemberType] # Clear previous pixmap


        # If QImage was successfully created and is valid
        # Use isNull() method on QImage object, check qt_image is not None first.
        # Check is needed runtime, ignore diagnostic.
        # Diagnostic reportUnknownMemberType on isNull stub addressed by ignore
        if qt_image is not None and not qt_image.isNull(): # type: ignore[reportUnnecessaryComparison, reportUnknownMemberType]
            # Convert QImage to QPixmap
            # Diagnostic reportUnknownMemberType on fromImage stub addressed by ignore
            pixmap = QPixmap.fromImage(qt_image) # type: ignore[reportUnknownMemberType]
            # Get the size of the target QLabel
            # Diagnostic reportUnknownMemberType on size stub addressed by ignore
            label_size: QSize = target_label.size() # type: ignore[reportUnknownMemberType]

            # Scale the pixmap to fit the label while preserving aspect ratio
            # Check for valid label dimensions before scaling
            if label_size.width() > 0 and label_size.height() > 0:
                # Use scaled method. Add ignore for stub issues.
                # Diagnostic reportUnknownMemberType on scaled stub addressed by ignore
                return pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation) # type: ignore[reportUnknownMemberType]

            # If label size is invalid, return the original pixmap (or None if pixmap creation failed)
            return pixmap

        # If QImage creation failed or resulted in a null/invalid image
        # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
        target_label.setText("QImage Null/Invalid"); # type: ignore[reportUnknownMemberType]
        # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
        target_label.setPixmap(QPixmap()); return None # type: ignore[reportUnknownMemberType] # Clear previous pixmap


    def process_and_display_frame_slot(self) -> None:
        """
        Processes the current frame through the pipeline and updates the display labels.
        This slot is called when a new frame is received or when pipeline parameters change.
        """
        # Do nothing if there is no current frame
        # self.current_cv_frame is Optional[NDArrayUInt8]. Check `is None` is correct. Ignore diagnostic.
        if self.current_cv_frame is None: # type: ignore[reportUnnecessaryComparison]
            # Ensure all labels are set to indicate no source or processing
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            self.original_video_label.setText("No Source"); # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            self.video_display_label.setText("No Source") # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            self.stage_input_video_label.setText("N/A") # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            self.stage_output_video_label.setText("N/A") # type: ignore[reportUnknownMemberType]
            # Clear any previously displayed pixmaps
            # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
            self.original_video_label.setPixmap(QPixmap()) # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
            self.video_display_label.setPixmap(QPixmap()) # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
            self.stage_input_video_label.setPixmap(QPixmap()) # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
            self.stage_output_video_label.setPixmap(QPixmap()) # type: ignore[reportUnknownMemberType]

            return

        # Start with a copy of the raw frame for pipeline processing
        # self.current_cv_frame is NDArrayUInt8 (subtype of CvMatLike)
        processed_frame_for_pipeline: CvMatLike = self.current_cv_frame.copy()

        # Initialize intermediate stage frames as None
        self.selected_stage_input_frame = None
        self.selected_stage_output_frame = None

        # Determine the ID of the currently selected operation for stage view
        # Diagnostic reportUnknownMemberType on currentItem stub addressed by ignore
        current_selected_list_item: Optional[QListWidgetItem] = self.operations_list_widget.currentItem() # type: ignore[reportUnknownMemberType]
        selected_op_id_for_stage_view: Optional[uuid.UUID] = None
        # Check if the selected item is our custom type. Check is necessary runtime logic. Ignore diagnostic.
        # Check is needed runtime, ignore diagnostic.
        if isinstance(current_selected_list_item, OperationListItem): # type: ignore[reportUnnecessaryComparison]
            selected_op_id_for_stage_view = current_selected_list_item.operation.id

        # Convert and display the original frame
        # self.original_frame_for_display is CaptureFrameType (NDArrayUInt8)
        # Check if original_frame_for_display is not None before passing it. Keep check, ignore diagnostic.
        # Check is needed runtime, ignore diagnostic.
        if self.original_frame_for_display is not None: # type: ignore[reportUnnecessaryComparison]
            original_pixmap = self._convert_cv_to_pixmap(self.original_frame_for_display, self.original_video_label)
            # _convert_cv_to_pixmap sets text on failure/None, no need for else here
            # Check is needed runtime, ignore diagnostic.
            if original_pixmap: # type: ignore[reportUnnecessaryComparison] # Set pixmap only if conversion was successful
                 # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
                 self.original_video_label.setPixmap(original_pixmap) # type: ignore[reportUnknownMemberType]
            # else: label text set by _convert_cv_to_pixmap on failure


        # Iterate through the pipeline operations using a temporary frame variable
        temp_frame_for_pipeline: CvMatLike = processed_frame_for_pipeline.copy() # Use a working copy

        # Process each operation in the pipeline
        for op_obj in self.operations_data: # Iterate through the internal list of operations (ImageOperation objects)

            # Keep a copy of the frame *before* applying the current operation
            # Ensure frame_before_this_op is CvMatLike (ndarray)
            frame_before_this_op: CvMatLike = np.asarray(temp_frame_for_pipeline).copy()

            # If this is the selected operation, store the frame *before* it
            # Check is needed runtime, ignore diagnostic.
            if selected_op_id_for_stage_view is not None and op_obj.id == selected_op_id_for_stage_view: # type: ignore[reportUnnecessaryComparison]
                self.selected_stage_input_frame = frame_before_this_op

            # If the operation is enabled and has a function reference
            # Check op_obj.function_ref is not None. Keep check, ignore diagnostic.
            # Check is needed runtime, ignore diagnostic.
            if op_obj.is_enabled and op_obj.function_ref is not None: # type: ignore[reportUnnecessaryComparison]
                try:
                    # Get the parameters for the function (Dict[str, ParamValue])
                    params_for_func = op_obj.get_params_for_processing()

                    # Apply the processing function.
                    # The function takes CvMatLike and returns CvMatLike.
                    # The function_ref is typed as ProcessingFunction which is Callable, so the call is valid.
                    result_frame: CvMatLike = op_obj.function_ref(temp_frame_for_pipeline, params_for_func)

                    # Update the temporary frame variable with the result.
                    # Ensure the result is a valid CvMatLike (ndarray) before assigning.
                    # Functions are typed to return CvMatLike, but defensive check is good.
                    # Check result_frame is not None. Keep check, ignore diagnostic.
                    # Check is needed runtime, ignore diagnostic.
                    if result_frame is not None and np.asarray(result_frame).size > 0: # type: ignore[reportUnnecessaryComparison]
                         temp_frame_for_pipeline = result_frame
                    else:
                         # If the result is invalid/empty, revert to the frame before this op.
                         print(f"Warning: Operation '{op_obj.base_name}' returned None or empty image. Reverting to previous frame.")
                         temp_frame_for_pipeline = frame_before_this_op


                except Exception as e:
                    # Catch errors during processing of a single operation
                    print(f"Error processing operation '{op_obj.base_name}': {e}")
                    # On error, revert the frame to what it was before this operation.
                    # This prevents a single failing operation from breaking the whole pipeline.
                    temp_frame_for_pipeline = frame_before_this_op
                    # Optionally, disable the failing operation in the UI here or show an error indicator.
                    # For now, just print and continue with the previous frame.
                    # break # Decide whether to stop the pipeline processing chain on the first error

            # If this is the selected operation, store the frame *after* it
            # Check is needed runtime, ignore diagnostic.
            if selected_op_id_for_stage_view is not None and op_obj.id == selected_op_id_for_stage_view: # type: ignore[reportUnnecessaryComparison]
                # Store a copy of the frame *after* this operation (whether it ran or failed/was disabled)
                # Ensure it's CvMatLike (ndarray) before copying
                self.selected_stage_output_frame = np.asarray(temp_frame_for_pipeline).copy()

        # The final frame after processing all operations
        # Ensure final frame is CvMatLike (ndarray)
        final_processed_frame: CvMatLike = np.asarray(temp_frame_for_pipeline)

        # --- Display Intermediate and Final Frames ---

        # Display the stage input and output frames if an operation is selected
        # Explicit check for None
        if selected_op_id_for_stage_view is not None: # type: ignore[reportUnnecessaryComparison]
            # Convert and display the stage input frame
            # self.selected_stage_input_frame is Optional[CvMatLike]. Check is done inside _convert_cv_to_pixmap.
            stage_input_pixmap = self._convert_cv_to_pixmap(self.selected_stage_input_frame, self.stage_input_video_label)
            # Check is needed runtime, ignore diagnostic.
            if stage_input_pixmap: # type: ignore[reportUnnecessaryComparison] # Set pixmap only if conversion was successful
                 # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
                 self.stage_input_video_label.setPixmap(stage_input_pixmap) # type: ignore[reportUnknownMemberType]
            # else: label text set by _convert_cv_to_pixmap on failure

            # Convert and display the stage output frame
            # self.selected_stage_output_frame is Optional[CvMatLike]. Check is done inside _convert_cv_to_pixmap.
            stage_output_pixmap = self._convert_cv_to_pixmap(self.selected_stage_output_frame, self.stage_output_video_label)
            # Check is needed runtime, ignore diagnostic.
            if stage_output_pixmap: # type: ignore[reportUnnecessaryComparison] # Set pixmap only if conversion was successful
                 # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
                 self.stage_output_video_label.setPixmap(stage_output_pixmap) # type: ignore[reportUnknownMemberType]
            # else: label text set by _convert_cv_to_pixmap on failure
        else:
            # If no operation is selected, clear stage views
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            self.stage_input_video_label.setText("No Stage Selected") # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on setText stub addressed by ignore
            self.stage_output_video_label.setText("No Stage Selected") # type: ignore[reportUnknownMemberType]
            # Also clear the stored stage frames
            self.selected_stage_input_frame = None
            self.selected_stage_output_frame = None
            # Clear any previous pixmaps by setting None (handled by setPixmap)
            # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
            self.stage_input_video_label.setPixmap(QPixmap()) # type: ignore[reportUnknownMemberType]
            # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
            self.stage_output_video_label.setPixmap(QPixmap()) # type: ignore[reportUnknownMemberType]

        # Convert and display the final processed frame
        # final_processed_frame is CvMatLike (ndarray)
        final_pixmap = self._convert_cv_to_pixmap(final_processed_frame, self.video_display_label)
        # Check is needed runtime, ignore diagnostic.
        if final_pixmap: # type: ignore[reportUnnecessaryComparison]
            # Diagnostic reportUnknownMemberType on setPixmap stub addressed by ignore
            self.video_display_label.setPixmap(final_pixmap) # type: ignore[reportUnknownMemberType]
            # Reset stylesheet to default if it was showing an error state
            # Check if the current stylesheet contains the error indicator style
            # Diagnostic reportUnknownMemberType on styleSheet/setStyleSheet stub addressed by ignore
            if self.video_display_label and "border: 1px solid red;" in self.video_display_label.styleSheet(): # type: ignore[reportUnnecessaryComparison, reportUnknownMemberType]
                 self.video_display_label.setStyleSheet(self.get_stylesheet()) # type: ignore[reportUnknownMemberType]
        else:
             # If final pixmap conversion failed, display an error on the label
             # _convert_cv_to_pixmap already set an error text and cleared the pixmap.
             # We might want to apply the error stylesheet here if it was an input error.
             # For now, trust _convert_cv_to_pixmap to set text and handle the style if it was a conversion issue.
             pass # No explicit else branch needed here.


    @override
    # Signature must match the base class QWidget.resizeEvent(a0: QResizeEvent | None). Added type hint and ignore.
    # Diagnostic reportIncompatibleMethod override addressed by ignore
    def resizeEvent(self, a0: QResizeEvent | None) -> None: # type: ignore[override]
        """Handles window resize events to update displayed images."""
        # Call the base class implementation. Pass the event object if it's not None.
        # a0 should not be None in typical resize event calls from the event loop.
        # Call base method. Add ignore for stub issues.
        # Diagnostic reportUnknownMemberType on base call stub addressed by ignore
        super().resizeEvent(a0) # type: ignore[reportUnknownMemberType]

        # If there's image content to display, re-process/scale it to the new label sizes.
        # Checking if the labels exist and if there's frame data available.
        # Labels are non-Optional after init_ui. current_cv_frame/original_frame_for_display are Optional.
        # Check is needed runtime, ignore diagnostic.
        # Check is needed runtime, ignore diagnostic.
        if self.current_cv_frame is not None or self.original_frame_for_display is not None: # type: ignore[reportUnnecessaryComparison, reportUnnecessaryComparison]
            # Use QTimer.singleShot to avoid immediate recursive calls if resizing triggers more events.
            # This smooths resizing behavior by delaying processing until the event loop is idle.
            # Delay by 100ms. Standard method. Add ignore.
            # Cast the slot method to Callable[[], None] for type hint, ignore signal stub.
            # Diagnostic reportUnknownMemberType on singleShot stub addressed by ignore
            QTimer.singleShot(100, cast(Callable[[], None], self.process_and_display_frame_slot)) # type: ignore[reportUnknownMemberType]


    @override
    # Signature must match the base class QWidget.closeEvent(a0: QCloseEvent | None). Added type hint and ignore.
    # Diagnostic reportIncompatibleMethod override addressed by ignore
    def closeEvent(self, a0: QCloseEvent | None) -> None: # type: ignore[override]
        """Handles application close events, ensuring video thread is stopped and connections are cleaned."""
        print("Closing application...")
        # Signal the video thread to stop and wait for it to finish.
        # The thread is parented to `self`, so its deletion should happen via `deleteLater`
        # and automatic cleanup when the parent window is deleted.
        # Explicit stop and wait is still good practice for graceful shutdown.
        # Check if video_thread is not None before accessing methods. Keep check, ignore diagnostic.
        # Diagnostic reportUnknownMemberType on isRunning stub addressed by ignore
        if self.video_thread is not None and self.video_thread.isRunning(): # type: ignore[reportUnnecessaryComparison, reportUnknownMemberType]
            print("Stopping video thread...")
            self.video_thread.stop()
            # Wait for up to 2 seconds for the thread to terminate. Standard method. Add ignore.
            # Diagnostic reportUnknownMemberType on wait stub addressed by ignore
            if not self.video_thread.wait(2000): # type: ignore[reportUnknownMemberType]
                print("Warning: Video thread did not stop gracefully within timeout.")
        # The thread object will be deleted when the parent window is deleted due to parentage.

        # Disconnect all connections stored in self._connections.
        # This is important to prevent calls to slots on deleted main window widgets
        # from signals emitted by objects (like list item emitters, thread signals)
        # that might be deleted later or in a different order.
        print(f"Disconnecting {len(self._connections)} signal connections...")
        connections_to_clear = list(self._connections) # Create a copy to iterate
        for conn_obj in connections_to_clear: # Iterate over the copy
            try:
                # Attempt to disconnect the connection object.
                # QObject.disconnect(conn_obj) disconnects the specific connection.
                # This might raise if the underlying sender/receiver is already gone.
                # This is the correct way to use the stored Connection object for disconnection.
                # Diagnostic reportUnknownMemberType on disconnect stub addressed by ignore
                QObject.disconnect(conn_obj) # type: ignore[reportUnknownMemberType]
                # No need to remove from list during iteration.
            except (TypeError, RuntimeError):
                 pass # Ignore errors during cleanup
        # Clear the original list after attempting all disconnections.
        self._connections.clear()
        print("Signal connections cleanup attempted.")

        # Accept the close event, allowing the application to exit.
        # Check if the event object is not None before calling accept().
        # a0 should not be None in typical close event calls from the event loop.
        # Call base implementation, which includes a0.accept() if not ignored.
        # Diagnostic reportUnknownMemberType on base call stub addressed by ignore
        super().closeEvent(a0) # type: ignore[reportUnknownMemberType]


    def save_pipeline_to_file(self) -> None:
        """Saves the current pipeline configuration to a JSON file."""
        # Ensure the internal operations_data is up-to-date with the list widget.
        self._update_operations_data_from_widget()

        # Create a list of dictionaries from the operations data.
        # Each op.to_dict() returns Dict[str, Any].
        pipeline_state: List[Dict[str, Any]] = [op.to_dict() for op in self.operations_data]

        # Wrap the pipeline state in a root dictionary (allows future expansion).
        full_save_data: Dict[str, Any] = {"pipeline": pipeline_state, "version": 1.0} # Add a version key

        # Open a file dialog to get the save path from the user.
        # QFileDialog.getSaveFileName returns tuple[str, str]. Standard method. Add ignore.
        # Diagnostic reportUnknownMemberType on getSaveFileName stub addressed by ignore
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Pipeline", "", "JSON Files (*.json)") # type: ignore[reportUnknownMemberType]

        if filePath: # If user selected a file path and is not an empty string
            # Ensure the file extension is .json if not provided
            if not filePath.lower().endswith('.json'):
                filePath += '.json'

            try:
                # Write the data to the file in JSON format with indentation.
                # Use default=str for UUID serialization.
                with open(filePath, 'w') as f:
                    json.dump(full_save_data, f, indent=4, default=str)
                # Diagnostic reportUnknownMemberType on information stub addressed by ignore
                QMessageBox.information(self, "Save Pipeline", "Pipeline saved successfully.") # type: ignore[reportUnknownMemberType]
            except Exception as e:
                # Diagnostic reportUnknownMemberType on critical stub addressed by ignore
                QMessageBox.critical(self, "Save Error", f"Could not save pipeline:\n{e}") # type: ignore[reportUnknownMemberType]

    def load_pipeline_from_file(self) -> None:
        """Loads a pipeline configuration from a JSON file."""
        # Open a file dialog to get the load path from the user.
        # QFileDialog.getOpenFileName returns tuple[str, str]. Standard method. Add ignore.
        # Diagnostic reportUnknownMemberType on getOpenFileName stub addressed by ignore
        filePath, _ = QFileDialog.getOpenFileName(self, "Load Pipeline", "", "JSON Files (*.json)") # type: ignore[reportUnknownMemberType]

        if filePath: # If user selected a file path and is not an empty string
            try:
                # Read and parse the JSON data from the file.
                with open(filePath, 'r') as f:
                    # JSON load result type is initially Any
                    loaded_data_any: Any = json.load(f)

                # Validate the loaded data structure. Necessary because source is Any. Keep check, ignore diagnostic.
                # Check is needed runtime, ignore diagnostic.
                if not isinstance(loaded_data_any, dict): # type: ignore[reportUnnecessaryIsInstance]
                    raise ValueError("Loaded JSON root is not a dictionary.")

                # Type is narrowed to dict[Any, Any] here by the isinstance check.
                # Use cast to assert the expected dictionary structure.
                loaded_data: Dict[str, Any] = cast(Dict[str, Any], loaded_data_any)

                # Get the pipeline state list.
                pipeline_state_any: Any = loaded_data.get("pipeline")
                # Validate the pipeline state list. Necessary because source is Any. Keep check, ignore diagnostic.
                # Check is needed runtime, ignore diagnostic.
                if not isinstance(pipeline_state_any, list): # type: ignore[reportUnnecessaryIsInstance]
                    raise ValueError("Pipeline data in JSON is not a list or is missing 'pipeline' key.")

                # Type is narrowed to list[Any]. Iterate directly over this list.
                # The `pipeline_state` variable assignment is removed as it was unused.

                # Clear the current pipeline UI and internal data.
                # Before clearing, manually disconnect signals that we stored connections for.
                print(f"Disconnecting {len(self._connections)} signal connections before loading...")
                connections_to_clear = list(self._connections) # Create a copy to iterate
                for conn_obj in connections_to_clear: # Iterate over the copy
                    try:
                        # Attempt to disconnect the connection object.
                        # Diagnostic reportUnknownMemberType on disconnect stub addressed by ignore
                        QObject.disconnect(conn_obj) # type: ignore[reportUnknownMemberType]
                        # No need to remove from list during iteration.
                    except (TypeError, RuntimeError):
                         pass # Ignore errors during cleanup

                # Clear the original list after attempting disconnection for all.
                self._connections.clear()
                print("Signal connections cleanup attempted before loading.")

                # Clear the list widget. This deletes the items and their child widgets/emitters.
                # Diagnostic reportUnknownMemberType on clear stub addressed by ignore
                self.operations_list_widget.clear() # type: ignore[reportUnknownMemberType]

                # Clear the internal list of operation objects
                self.operations_data.clear()

                # Clear the parameter panel display and stored stage frames.
                # This also removes parameter widgets and their connections managed by the panel's layout.
                self.on_operation_selected(None, None)

                # Create ImageOperation instances from the loaded data
                # Iterate over the list (type list[Any] after isinstance check)
                for op_data_item_any in pipeline_state_any: # type: ignore[reportUnknownVariableType] # Ignore the 'Any' type of the loop variable itself
                    # Validate each item in the list. Necessary because source is Any. Keep check, ignore diagnostic.
                    # Check is needed runtime, ignore diagnostic.
                    if isinstance(op_data_item_any, dict): # type: ignore[reportUnnecessaryIsInstance]
                        # Type of op_data_item_any is narrowed to dict[Any, Any]. Cast for clarity before using.
                        op_data_item: Dict[str, Any] = cast(Dict[str, Any], op_data_item_any) # Cast after successful check
                        try:
                            # Create an ImageOperation instance from the dictionary data using the class method
                            # Use the casted op_data_item dictionary
                            loaded_op_instance: ImageOperation = ImageOperation.from_dict(op_data_item, self.processing_functions)
                            # Add the existing ImageOperation object to the UI list widget
                            # Use the dedicated helper function that adds an *existing* operation instance.
                            self._add_operation_to_list_widget_item(loaded_op_instance)
                        except Exception as e:
                            # Log the raw item data for debugging if it failed
                            print(f"Error creating ImageOperation from loaded data: {op_data_item_any}. Skipping. Error: {e}")
                    # Check is needed runtime, ignore diagnostic.
                    else: # type: ignore[reportUnnecessaryComparison]
                        print(f"Warning: Unexpected item format in pipeline list during load: {op_data_item_any}. Skipping.")

                # Update the internal operations data list from the widget after adding items.
                # This call also updates the numbering labels on the items.
                self._update_operations_data_from_widget()

                # Select the first item if the list is not empty, triggering parameter panel update.
                # Diagnostic reportUnknownMemberType on count stub addressed by ignore
                if self.operations_list_widget.count() > 0: # type: ignore[reportUnknownMemberType]
                    # Diagnostic reportUnknownMemberType on setCurrentRow stub addressed by ignore
                    self.operations_list_widget.setCurrentRow(0) # type: ignore[reportUnknownMemberType]

                # Re-process and display the current frame with the loaded pipeline.
                # Use singleShot to allow UI to update after loading items. Standard method. Add ignore.
                # Cast the slot method to Callable[[], None] for type hint, ignore signal stub.
                # Diagnostic reportUnknownMemberType on singleShot stub addressed by ignore
                QTimer.singleShot(10, cast(Callable[[], None], self.process_and_display_frame_slot)) # type: ignore[reportUnknownMemberType]

                # Diagnostic reportUnknownMemberType on information stub addressed by ignore
                QMessageBox.information(self, "Load Pipeline", "Pipeline loaded successfully.") # type: ignore[reportUnknownMemberType]
                # Optionally check and report pipeline version differences if needed
                loaded_version = loaded_data.get("version", 0)
                if loaded_version > 1.0: # Assuming 1.0 is current version
                     print(f"Warning: Loaded pipeline version {loaded_version} is newer than expected (1.0). Compatibility issues may occur.")


            except (FileNotFoundError, json.JSONDecodeError, ValueError, Exception) as e:
                # Catch specific file/JSON/value errors and generic exceptions
                # Diagnostic reportUnknownMemberType on critical stub addressed by ignore
                QMessageBox.critical(self, "Load Error", f"Could not load pipeline:\n{e}") # type: ignore[reportUnknownMemberType]


# --- Main Application Entry Point ---
def main():
    # Create a QApplication instance (necessary for any PyQt6 application)
    # Diagnostic reportUnknownMemberType on QApplication stub addressed by ignore
    app = QApplication(sys.argv) # type: ignore[reportUnknownMemberType]

    # Create the main window instance
    # Optionally pass a command line argument for video source (e.g., camera ID or file path)
    initial_source: Union[int, str] = 0 # Default to camera 0
    if len(sys.argv) > 1:
        source_arg = sys.argv[1]
        try:
            # Try converting argument to an integer (camera ID)
            initial_source = int(source_arg)
        except ValueError:
            # If conversion to int fails, treat it as a string (file path or URL)
            initial_source = source_arg

    main_window = ProcessingPipelineApp(video_source_arg=initial_source)

    # Show the main window
    # Diagnostic reportUnknownMemberType on show stub addressed by ignore
    main_window.show() # type: ignore[reportUnknownMemberType]

    # Start the application's event loop. sys.exit ensures a clean exit.
    # Diagnostic reportUnknownMemberType on exec stub addressed by ignore
    sys.exit(app.exec()) # type: ignore[reportUnknownMemberType]

if __name__ == "__main__":
    main() # Call the new main function