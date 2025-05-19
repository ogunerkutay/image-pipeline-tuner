#test_qt_pipeline_gui.py

import unittest
from unittest.mock import patch, MagicMock, ANY
import numpy as np
import uuid
from typing import Dict, Any, Callable, Union # Keep necessary typing imports


# Attempt to import cv2, and if it fails, set up a mock for it.
try:
    import cv2
except ImportError:
    # Use Union to hint that cv2 might be the real module or the mock
    # Add type: ignore because MagicMock doesn't fully implement the cv2 module interface type-wise
    cv2: Union[type(cv2), MagicMock] = MagicMock() # type: ignore
    # Mock common cv2 attributes and methods used if cv2 is not installed
    cv2.COLOR_BGR2GRAY = 1
    cv2.COLOR_BGRA2GRAY = 2
    cv2.CV_8U = 0
    cv2.NORM_MINMAX = 32
    # Ensured mock return values have correct runtime dtypes, rely on inference
    cv2.GaussianBlur = MagicMock(return_value=np.array([[0]], dtype=np.uint8))
    cv2.cvtColor = MagicMock(return_value=np.array([[0]], dtype=np.uint8))
    cv2.bilateralFilter = MagicMock(return_value=np.array([[0]], dtype=np.uint8))
    cv2.createCLAHE = MagicMock()
    cv2.adaptiveThreshold = MagicMock(return_value=np.array([[0]], dtype=np.uint8))
    cv2.Canny = MagicMock(return_value=np.array([[0]], dtype=np.uint8))
    cv2.HoughCircles = MagicMock(return_value=None)
    cv2.findContours = MagicMock(return_value=([], None))
    cv2.drawContours = MagicMock(return_value=np.array([[0]], dtype=np.uint8))
    cv2.morphologyEx = MagicMock(return_value=np.array([[0]], dtype=np.uint8))
    cv2.undistort = MagicMock(return_value=np.array([[0]], dtype=np.uint8))
    cv2.resize = MagicMock(return_value=np.array([[0]], dtype=np.uint8))
    # Mock getOptimalNewCameraMatrix return values - return types are inferred by np.eye
    cv2.getOptimalNewCameraMatrix = MagicMock(return_value=(np.eye(3, dtype=np.float64), (0,0,10,10)))
    # Mock initUndistortRectifyMap return values - return types are inferred by np.array
    cv2.initUndistortRectifyMap = MagicMock(return_value=(np.array([[]], dtype=np.float32), np.array([[]], dtype=np.float32)))
    cv2.remap = MagicMock(return_value=np.array([[0]], dtype=np.uint8))
    cv2.threshold = MagicMock(return_value=(0.0, np.array([[0]], dtype=np.uint8)))
    cv2.boundingRect = MagicMock(return_value=(0,0,10,10))
    cv2.contourArea = MagicMock(return_value=100.0)
    cv2.arcLength = MagicMock(return_value=10.0)
    cv2.getStructuringElement = MagicMock(return_value=np.ones((3,3), dtype=np.uint8))


from image_pipeline_tuner import (
    to_int_robust,
    to_float_robust,
    _ensure_uint8_image,       # Returns CvMatLike | None
    _ensure_grayscale_uint8,   # Returns NDArrayUInt8 | None (assuming this is its defined return type)
    ImageOperation,
    ProcessingPipelineApp, # For accessing processing_functions and configs
    apply_grayscale,           # Returns CvMatLike | None
    apply_gaussian_blur,       # Returns CvMatLike | None
    ParamConfigValue,
    CvMatLike,                 # Likely Union[NDArrayUInt8, NDArrayFloat32, ...] or similar
    NDArrayUInt8,              # Custom alias from realtime_video_pipeline_qt_gui
    NDArrayFloat32,            # Custom alias from realtime_video_pipeline_qt_gui
    ParamValue                 # Needed for typing the params dict in tests
)

# Removed local aliases for specific numpy types


class TestTypeConversionUtils(unittest.TestCase):
    def test_to_int_robust(self):
        self.assertEqual(to_int_robust(10), 10)
        self.assertEqual(to_int_robust(10.7), 10)
        self.assertEqual(to_int_robust("15"), 15)
        self.assertEqual(to_int_robust("abc", default=5), 5)
        self.assertEqual(to_int_robust(None, default=-1), -1)
        self.assertEqual(to_int_robust(True), 1) # True converts to 1
        self.assertEqual(to_int_robust(False), 0) # False converts to 0

    def test_to_float_robust(self):
        self.assertEqual(to_float_robust(10.5), 10.5)
        self.assertEqual(to_float_robust(10), 10.0)
        self.assertEqual(to_float_robust("15.3"), 15.3)
        self.assertEqual(to_float_robust("abc", default=5.1), 5.1)
        self.assertEqual(to_float_robust(None, default=-1.2), -1.2)
        self.assertEqual(to_float_robust(True), 1.0)
        self.assertEqual(to_float_robust(False), 0.0)

class TestImageProcessingHelpers(unittest.TestCase):
    def test_ensure_uint8_image(self):
        # Test with uint8 input
        # Remove explicit hint, rely on inference
        img_uint8 = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        # Remove explicit hint, rely on inference (type is CvMatLike | None from function signature)
        result = _ensure_uint8_image(img_uint8)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertTrue(np.array_equal(result, img_uint8))
            self.assertEqual(result.dtype, np.uint8)

        # Test with float input [0, 1]
        # Remove explicit hint, rely on inference
        img_float = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        # Remove explicit hint on local variable created directly with np.array, rely on inference
        expected_float_to_uint8 = np.array([[25, 51], [76, 102]], dtype=np.uint8)
        # Remove explicit hint, rely on inference (type is CvMatLike | None from function signature)
        result = _ensure_uint8_image(img_float)
        self.assertIsNotNone(result)
        if result is not None:
             self.assertTrue(np.array_equal(result, expected_float_to_uint8))
             self.assertEqual(result.dtype, np.uint8)

        # Test with float input outside [0, 1] (should normalize)
        # Remove explicit hint, rely on inference
        img_float_large = np.array([[0, 1000], [500, 2000]], dtype=np.float32)
        with patch('cv2.normalize') as mock_normalize:
            # Mock normalize return type inference handles this
            mock_normalize.return_value = np.zeros(img_float_large.shape, dtype=np.uint8)
            # Remove explicit hint, rely on inference (type is CvMatLike | None from function signature)
            result = _ensure_uint8_image(img_float_large)
            mock_normalize.assert_called_once()
            self.assertIsNotNone(result)
            if result is not None:
                self.assertEqual(result.dtype, np.uint8)
                self.assertEqual(result.shape, img_float_large.shape)

        # Test with uint16 input (should normalize)
        # Remove explicit hint on local variable, rely on inference
        img_uint16 = np.array([[0, 65535], [10000, 30000]], dtype=np.uint16)
        with patch('cv2.normalize') as mock_normalize:
             # Mock normalize return type inference handles this
             mock_normalize.return_value = np.zeros(img_uint16.shape, dtype=np.uint8)
             # Remove explicit hint, rely on inference (type is CvMatLike | None from function signature)
             result = _ensure_uint8_image(img_uint16)
             mock_normalize.assert_called_once()
             self.assertIsNotNone(result)
             if result is not None:
                 self.assertEqual(result.dtype, np.uint8)
                 self.assertEqual(result.shape, img_uint16.shape)


    def test_ensure_grayscale_uint8(self):
        # Test with already grayscale uint8
        # Remove explicit hint on local variable, rely on inference
        img_gray_uint8 = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        # Remove explicit hint, rely on inference (type is NDArrayUInt8 | None from function signature)
        result = _ensure_grayscale_uint8(img_gray_uint8)

        self.assertIsNotNone(result)
        if result is not None:
            # Use np.array_equal which accepts array-like, inference is sufficient for args
            self.assertTrue(np.array_equal(result, img_gray_uint8))
            self.assertEqual(result.dtype, np.uint8)
            self.assertEqual(result.ndim, 2)

        # Test with BGR uint8
        # Remove explicit hint on local variable, rely on inference
        img_bgr_uint8 = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        with patch('cv2.cvtColor') as mock_cvtcolor:
            # Mock return value type inference handles this
            mock_cvtcolor.return_value = np.zeros((10,10), dtype=np.uint8)
            # Remove explicit hint, rely on inference (type is NDArrayUInt8 | None from function signature)
            result = _ensure_grayscale_uint8(img_bgr_uint8)
            mock_cvtcolor.assert_called_with(ANY, cv2.COLOR_BGR2GRAY)
            self.assertIsNotNone(result)
            if result is not None: # Pylance narrowing
                self.assertEqual(result.dtype, np.uint8)
                self.assertEqual(result.ndim, 2)

        # Test with BGRA uint8
        # Remove explicit hint on local variable, rely on inference
        img_bgra_uint8 = np.random.randint(0, 256, (10, 10, 4), dtype=np.uint8)
        with patch('cv2.cvtColor') as mock_cvtcolor:
            # Mock return value type inference handles this
            mock_cvtcolor.return_value = np.zeros((10,10), dtype=np.uint8)
            # Remove explicit hint, rely on inference (type is NDArrayUInt8 | None from function signature)
            result = _ensure_grayscale_uint8(img_bgra_uint8)
            mock_cvtcolor.assert_called_with(ANY, cv2.COLOR_BGRA2GRAY)
            self.assertIsNotNone(result)
            if result is not None: # Pylance narrowing
                self.assertEqual(result.dtype, np.uint8)
                self.assertEqual(result.ndim, 2)

        # Test with BGR float32 (needs uint8 conversion first)
        # Remove explicit hint on local variable, rely on inference
        img_bgr_float32 = np.random.rand(10, 10, 3).astype(np.float32)
        with patch('cv2.cvtColor') as mock_cvtcolor, \
             patch('realtime_video_pipeline_qt_gui._ensure_uint8_image') as mock_ensure_uint8:
            mock_ensure_uint8.side_effect = lambda x: x.astype(np.uint8) if x.dtype != np.uint8 else x.copy()
            # mock_cvtcolor returns NDArrayUInt8, type inference handles this
            mock_cvtcolor.return_value = np.zeros((10,10), dtype=np.uint8)

            # Remove explicit hint, rely on inference (type is NDArrayUInt8 | None from function signature)
            result = _ensure_grayscale_uint8(img_bgr_float32)

            self.assertGreaterEqual(mock_ensure_uint8.call_count, 1)
            mock_cvtcolor.assert_called_with(ANY, cv2.COLOR_BGR2GRAY)
            self.assertIsNotNone(result)
            if result is not None: # Pylance narrowing
                self.assertEqual(result.dtype, np.uint8)
                self.assertEqual(result.ndim, 2)

        # Test with unsupported shape (e.g., 1D)
        # Remove explicit hint on 1D array, rely on inference
        img_1d = np.array([1, 2, 3], dtype=np.uint8)
        # Remove explicit hint, rely on inference (type is NDArrayUInt8 | None from function signature)
        result = _ensure_grayscale_uint8(img_1d) # Pass the 1D array to the function
        self.assertIsNone(result) # Explicitly test the None return case


class TestImageOperation(unittest.TestCase):
    def setUp(self):
        self.default_params: Dict[str, ParamConfigValue] = {
            "param1": {"type": "int", "value": 10, "min": 0, "max": 100},
            "param2": {"type": "float", "value": 0.5, "min": 0.0, "max": 1.0}
        }
        # Use the Callable type alias
        ProcessingFunctionType = Callable[[CvMatLike, Dict[str, ParamValue]], CvMatLike | None]
        # Remove explicit hint on the instance variable, rely on inference from MagicMock + return_value
        self.mock_func = MagicMock(return_value=np.array([[0]], dtype=np.uint8))
        # Type the dictionary using the Callable alias
        self.processing_functions_map: Dict[str, ProcessingFunctionType] = {"TestOp": self.mock_func}


    def test_image_operation_init(self):
        op = ImageOperation(base_name="MyTestOp",
                            op_type_name="TestOp",
                            # function_ref parameter is hinted in ImageOperation.__init__
                            function_ref=self.mock_func,
                            default_params=self.default_params,
                            is_enabled=True)
        self.assertIsInstance(op.id, uuid.UUID)
        self.assertEqual(op.base_name, "MyTestOp")
        self.assertEqual(op.op_type_name, "TestOp")
        self.assertEqual(op.function_ref, self.mock_func)
        self.assertEqual(op.params["param1"]["value"], 10)
        self.assertNotEqual(id(op.params), id(self.default_params)) # Should be a copy
        self.assertNotEqual(id(op.params["param1"]), id(self.default_params["param1"])) # Inner dicts should be copied
        self.assertTrue(op.is_enabled)

    def test_get_params_for_processing(self):
        op = ImageOperation(base_name="Test", default_params=self.default_params)
        # Keep explicit hint on the dictionary return
        params_for_processing: Dict[str, ParamValue] = op.get_params_for_processing()
        expected_params: Dict[str, ParamValue] = {"param1": 10, "param2": 0.5}
        self.assertEqual(params_for_processing, expected_params)

    def test_to_dict_serialization(self):
        op = ImageOperation(base_name="MyTestOp",
                            op_type_name="TestOp",
                            default_params=self.default_params,
                            is_enabled=False)
        op_id_str = str(op.id) # Capture ID for comparison
        op_dict: Dict[str, Any] = op.to_dict() # Explicitly type

        self.assertEqual(op_dict["id"], op_id_str)
        self.assertEqual(op_dict["base_name"], "MyTestOp")
        self.assertEqual(op_dict["op_type_name"], "TestOp")
        # Access value safely (although to_dict should guarantee structure)
        self.assertIn("params", op_dict)
        if "params" in op_dict and isinstance(op_dict["params"], dict):
            self.assertIn("param1", op_dict["params"])
            if "param1" in op_dict["params"] and isinstance(op_dict["params"]["param1"], dict):
                 self.assertIn("value", op_dict["params"]["param1"])
                 self.assertEqual(op_dict["params"]["param1"]["value"], 10)
        self.assertFalse(op_dict["is_enabled"])

    def test_from_dict_deserialization_valid(self):
        op_id = uuid.uuid4()
        data: Dict[str, Any] = {
            "id": str(op_id),
            "base_name": "LoadedOp",
            "op_type_name": "TestOp",
            "params": {
                "param1": {"type": "int", "value": 20, "min": 0, "max": 50},
                "param_new": {"type": "text", "value": "hello"}
            },
            "is_enabled": True
        }
        # Remove explicit hint, rely on inference (type is ImageOperation | None from function signature)
        op = ImageOperation.from_dict(data, self.processing_functions_map)
        self.assertEqual(op.id, op_id)
        self.assertEqual(op.base_name, "LoadedOp")
        self.assertEqual(op.op_type_name, "TestOp")
        self.assertEqual(op.function_ref, self.mock_func)
        self.assertIn("param1", op.params)
        if "param1" in op.params and isinstance(op.params["param1"], dict):
            self.assertIn("value", op.params["param1"])
            self.assertEqual(op.params["param1"]["value"], 20)
        self.assertIn("param_new", op.params)
        if "param_new" in op.params and isinstance(op.params["param_new"], dict):
            self.assertIn("value", op.params["param_new"])
            self.assertEqual(op.params["param_new"]["value"], "hello")
        self.assertTrue(op.is_enabled)

    def test_from_dict_missing_op_type_name_uses_base_name(self):
        # Make a processing function available for "FallbackOp"
        ProcessingFunctionType = Callable[[CvMatLike, Dict[str, ParamValue]], CvMatLike | None]
        # Remove explicit hint on local variable, rely on inference
        fallback_mock_func = MagicMock(return_value=np.array([[0]], dtype=np.uint8))
        processing_funcs: Dict[str, ProcessingFunctionType] = {"FallbackOp": fallback_mock_func}
        data: Dict[str, Any] = {
            "base_name": "FallbackOp",
            "params": {},
            "is_enabled": True
        }
        # Remove explicit hint, rely on inference (type is ImageOperation | None from function signature)
        op = ImageOperation.from_dict(data, processing_funcs)
        self.assertEqual(op.op_type_name, "FallbackOp")
        self.assertEqual(op.function_ref, fallback_mock_func)

    def test_from_dict_unknown_op_type_name(self):
        data: Dict[str, Any] = {
            "base_name": "UnknownOp",
            "op_type_name": "NonExistentOp",
            "params": {}, "is_enabled": True
        }
        # Remove explicit hint, rely on inference (type is ImageOperation | None from function signature)
        op = ImageOperation.from_dict(data, self.processing_functions_map)
        self.assertIsNone(op.function_ref)

    def test_from_dict_invalid_params_structure(self):
        data: Dict[str, Any] = {
            "base_name": "OpWithBadParams",
            "op_type_name": "TestOp",
            "params": {"param1": "not_a_dict"},
            "is_enabled": True
        }
        # Remove explicit hint, rely on inference (type is ImageOperation | None from function signature)
        op = ImageOperation.from_dict(data, self.processing_functions_map)
        self.assertNotIn("param1", op.params)

        data_missing_type_value: Dict[str, Any] = {
            "base_name": "OpWithBadParams2",
            "op_type_name": "TestOp",
            "params": {"param1": {"min": 0}},
            "is_enabled": True
        }
        # Remove explicit hint, rely on inference (type is ImageOperation | None from function signature)
        op2 = ImageOperation.from_dict(data_missing_type_value, self.processing_functions_map)
        self.assertNotIn("param1", op2.params)

    def test_from_dict_default_id_generation(self):
        data: Dict[str, Any] = {
            "base_name": "OpWithoutId",
            "op_type_name": "TestOp",
            "params": {}, "is_enabled": True
        }
        # Remove explicit hint, rely on inference (type is ImageOperation | None from function signature)
        op = ImageOperation.from_dict(data, self.processing_functions_map)
        self.assertIsInstance(op.id, uuid.UUID)


class TestCoreProcessingFunctions(unittest.TestCase):

    def setUp(self):
        # Remove explicit hints on instance variables, rely on inference
        self.bgr_image_uint8 = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        self.gray_image_uint8 = np.random.randint(0, 256, (10, 10), dtype=np.uint8)

    @patch('realtime_video_pipeline_qt_gui._ensure_grayscale_uint8')
    def test_apply_grayscale(self, mock_ensure_grayscale: MagicMock):
        # Mock return value inference handles this
        mock_ensure_grayscale.return_value = self.gray_image_uint8.copy()

        # Remove explicit hint, rely on inference (type is CvMatLike | None from function signature)
        result = apply_grayscale(self.bgr_image_uint8, {})

        mock_ensure_grayscale.assert_called_once_with(self.bgr_image_uint8)
        self.assertIsNotNone(result)
        if result is not None:
             self.assertTrue(np.array_equal(result, mock_ensure_grayscale.return_value))

    @patch('realtime_video_pipeline_qt_gui._ensure_grayscale_uint8')
    def test_apply_grayscale_conversion_fails(self, mock_ensure_grayscale: MagicMock):
        mock_ensure_grayscale.return_value = None
        original_copy = self.bgr_image_uint8.copy()

        # Remove explicit hint, rely on inference (type is CvMatLike | None from function signature)
        result = apply_grayscale(self.bgr_image_uint8, {})

        mock_ensure_grayscale.assert_called_once_with(self.bgr_image_uint8)
        self.assertIsNotNone(result)
        if result is not None:
             self.assertTrue(np.array_equal(result, original_copy))
             self.assertNotEqual(id(result), id(self.bgr_image_uint8))


    @patch('cv2.GaussianBlur')
    def test_apply_gaussian_blur(self, mock_cv_gaussian_blur: MagicMock):
        # Mock return value inference handles this
        mock_cv_gaussian_blur.return_value = self.bgr_image_uint8.copy()
        params: Dict[str, ParamValue] = {
            "ksize_w": 5, "ksize_h": 5, "sigmaX": 1.0, "sigmaY": 1.0
        }
        # Remove explicit hint, rely on inference (type is CvMatLike | None from function signature)
        result = apply_gaussian_blur(self.bgr_image_uint8, params)

        mock_cv_gaussian_blur.assert_called_once_with(
            self.bgr_image_uint8, (5, 5), sigmaX=1.0, sigmaY=1.0
        )
        self.assertIsNotNone(result)
        if result is not None:
            self.assertTrue(np.array_equal(result, mock_cv_gaussian_blur.return_value))

    def test_apply_gaussian_blur_kernel_validation(self):
        params_even_ksize: Dict[str, ParamValue] = {"ksize_w": 4, "ksize_h": 0, "sigmaX": 0, "sigmaY": 0}
        with patch('cv2.GaussianBlur') as mock_cv_blur:
            mock_cv_blur.return_value = self.bgr_image_uint8.copy()
            # No need for type hint on the call result if not used explicitly typed
            apply_gaussian_blur(self.bgr_image_uint8, params_even_ksize)
            mock_cv_blur.assert_called_with(ANY, (5, 1), sigmaX=0, sigmaY=0)

    @patch('cv2.bilateralFilter')
    @patch('realtime_video_pipeline_qt_gui._ensure_uint8_image')
    def test_apply_bilateral_filter(self, mock_ensure_uint8: MagicMock, mock_cv_bilateral: MagicMock):
         mock_ensure_uint8.return_value = self.bgr_image_uint8.copy()
         mock_cv_bilateral.return_value = self.bgr_image_uint8.copy()
         params: Dict[str, ParamValue] = {"d": 9, "sigmaColor": 75.0, "sigmaSpace": 75.0}

         ProcessingFunctionType = Callable[[CvMatLike, Dict[str, ParamValue]], CvMatLike | None]
         bilateral_func: ProcessingFunctionType = ProcessingPipelineApp.processing_functions["Bilateral Filter"]
         self.assertTrue(callable(bilateral_func))
         # Remove explicit hint, rely on inference (type is CvMatLike | None from function signature)
         result = bilateral_func(self.bgr_image_uint8, params)

         mock_ensure_uint8.assert_called_once_with(self.bgr_image_uint8)
         mock_cv_bilateral.assert_called_once_with(self.bgr_image_uint8, 9, 75.0, 75.0)
         self.assertIsNotNone(result)
         if result is not None:
              self.assertTrue(np.array_equal(result, mock_cv_bilateral.return_value))

    @patch('cv2.resize')
    def test_apply_resize(self, mock_cv_resize: MagicMock):
         target_dims = (320, 240)
         mock_cv_resize.return_value = np.zeros((*target_dims, 3), dtype=np.uint8)
         params: Dict[str, ParamValue] = {"width": target_dims[0], "height": target_dims[1], "interpolation": "linear"}

         ProcessingFunctionType = Callable[[CvMatLike, Dict[str, ParamValue]], CvMatLike | None]
         resize_func: ProcessingFunctionType = ProcessingPipelineApp.processing_functions["Resize"]
         self.assertTrue(callable(resize_func))
         # Remove explicit hint, rely on inference (type is CvMatLike | None from function signature)
         result = resize_func(self.bgr_image_uint8, params)

         mock_cv_resize.assert_called_once_with(self.bgr_image_uint8, target_dims, interpolation=cv2.INTER_LINEAR)
         self.assertIsNotNone(result)
         if result is not None:
             self.assertEqual(result.shape[:2], target_dims)

    @patch('cv2.remap')
    @patch('cv2.initUndistortRectifyMap')
    @patch('cv2.getOptimalNewCameraMatrix')
    def test_apply_undistort(self, mock_get_optimal_cam: MagicMock, mock_init_rect_map: MagicMock, mock_remap: MagicMock):
        # Mock return values - rely on numpy inference for local vars, remove explicit hints
        mock_optimal_k = np.eye(3, dtype=np.float64)
        mock_roi = (0, 0, self.bgr_image_uint8.shape[1], self.bgr_image_uint8.shape[0])
        mock_get_optimal_cam.return_value = (mock_optimal_k, mock_roi)

        # Mock initUndistortRectifyMap returns two float32 arrays - rely on numpy inference
        mock_mapx = np.random.rand(self.bgr_image_uint8.shape[0], self.bgr_image_uint8.shape[1]).astype(np.float32)
        mock_mapy = np.random.rand(self.bgr_image_uint8.shape[0], self.bgr_image_uint8.shape[1]).astype(np.float32)
        mock_init_rect_map.return_value = (mock_mapx, mock_mapy)

        # Mock remap should match expected output shape/dtype (likely NDArrayUInt8 based on test data)
        # Remove explicit hint on local variable, rely on inference
        undistorted_full = self.bgr_image_uint8.copy()
        mock_remap.return_value = undistorted_full

        params: Dict[str, ParamValue] = {
            "camera_matrix_str": "1000,0,320;0,1000,240;0,0,1",
            "dist_coeffs_str": "0.1,0.01,0,0,0",
            "alpha": 0.0
        }

        ProcessingFunctionType = Callable[[CvMatLike, Dict[str, ParamValue]], CvMatLike | None]
        undistort_func: ProcessingFunctionType = ProcessingPipelineApp.processing_functions["Undistort"]
        self.assertTrue(callable(undistort_func))
        # Remove explicit hint, rely on inference (type is CvMatLike | None from function signature)
        result = undistort_func(self.bgr_image_uint8, params)

        mock_get_optimal_cam.assert_called_once_with(ANY, ANY, ANY, ANY, ANY, ANY)
        mock_init_rect_map.assert_called_once_with(ANY, ANY, ANY, ANY, ANY, ANY)
        mock_remap.assert_called_once_with(self.bgr_image_uint8, ANY, ANY, cv2.INTER_LINEAR)

        self.assertIsNotNone(result)
        if result is not None:
            self.assertTrue(np.array_equal(result, undistorted_full))

    def test_apply_undistort_invalid_params(self):
        params_bad_k: Dict[str, ParamValue] = {"camera_matrix_str": "invalid", "dist_coeffs_str": "0,0,0,0,0", "alpha": 0.0}
        ProcessingFunctionType = Callable[[CvMatLike, Dict[str, ParamValue]], CvMatLike | None]
        undistort_func: ProcessingFunctionType = ProcessingPipelineApp.processing_functions["Undistort"]
        self.assertTrue(callable(undistort_func))
        # Remove explicit hint, rely on inference (type is CvMatLike | None from function signature)
        result = undistort_func(self.bgr_image_uint8, params_bad_k)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertTrue(np.array_equal(result, self.bgr_image_uint8))

        params_bad_d: Dict[str, ParamValue] = {"camera_matrix_str": "1,0,0;0,1,0;0,0,1", "dist_coeffs_str": "invalid", "alpha": 0.0}
        # Remove explicit hint, rely on inference (type is CvMatLike | None from function signature)
        result2 = undistort_func(self.bgr_image_uint8, params_bad_d)
        self.assertIsNotNone(result2)


class TestAppConfigurationConsistency(unittest.TestCase):
    def test_available_op_configs_and_processing_functions(self):
        app_instance = ProcessingPipelineApp
        for op_config in app_instance.available_op_configs:
            self.assertIn("op_type_name", op_config)
            op_type_name = op_config["op_type_name"]
            self.assertIn(op_type_name, app_instance.processing_functions,
                          f"Processing function for '{op_type_name}' is missing.")

    def test_operation_details_param_explanations_structure(self):
        app_instance = ProcessingPipelineApp
        for op_type_name, details_entry in app_instance.operation_details.items():
            self.assertIn("explanation", details_entry, f"'explanation' missing for '{op_type_name}'")
            if "explanation" in details_entry:
                self.assertIsInstance(details_entry["explanation"], str,
                                      f"'explanation' for '{op_type_name}' is not a string.")

            if "param_explanations" in details_entry:
                param_explanations = details_entry["param_explanations"]
                self.assertIsInstance(param_explanations, dict,
                                      f"'param_explanations' for '{op_type_name}' is not a dict.")
                if isinstance(param_explanations, dict):
                    for param_name, explanation in param_explanations.items():
                        self.assertIsInstance(param_name, str,
                                              f"Parameter name key in 'param_explanations' for '{op_type_name}' is not a string: {param_name}")
                        self.assertIsInstance(explanation, str,
                                              f"Explanation for param '{param_name}' in '{op_type_name}' is not a string: {explanation}")

    def test_param_consistency_between_available_and_details(self):
        app_instance = ProcessingPipelineApp
        available_op_params_map: Dict[str, set[str]] = {
            cfg["op_type_name"]: set(cfg["params"].keys())
            for cfg in app_instance.available_op_configs if "op_type_name" in cfg and "params" in cfg and isinstance(cfg["params"], dict)
        }

        for op_type_name, details_entry in app_instance.operation_details.items():
            if "param_explanations" in details_entry and isinstance(details_entry["param_explanations"], dict):
                param_explanations = details_entry["param_explanations"]
                explained_params = set(param_explanations.keys())

                self.assertIn(op_type_name, available_op_params_map,
                              f"'{op_type_name}' found in operation_details with param_explanations but not in available_op_configs for param checking.")

                defined_params = available_op_params_map[op_type_name]
                for explained_param_name in explained_params:
                    self.assertIn(explained_param_name, defined_params,
                                  f"Param '{explained_param_name}' in 'param_explanations' for '{op_type_name}' "
                                  f"is not a defined parameter in 'available_op_configs'. Defined: {defined_params}")

    def test_param_config_value_choices_type(self):
        app_instance = ProcessingPipelineApp
        for op_config in app_instance.available_op_configs:
            self.assertIn("params", op_config)
            if "params" in op_config and isinstance(op_config["params"], dict):
                 for param_name, param_detail in op_config["params"].items():
                    self.assertIn("type", param_detail)
                    if "type" in param_detail and param_detail["type"] == "choice":
                        self.assertIn("choices", param_detail,
                                      f"'choices' missing for 'choice' type param '{param_name}' in '{op_config.get('op_type_name', 'Unknown')}'.")
                        choices = param_detail.get("choices")
                        self.assertIsInstance(choices, dict,
                                              f"'choices' for param '{param_name}' in '{op_config.get('op_type_name', 'Unknown')}' is not a dict.")
                        if isinstance(choices, dict):
                            for choice_display, choice_value in choices.items():
                                self.assertIsInstance(choice_display, str,
                                                      f"Display text for a choice in param '{param_name}' ('{op_config.get('op_type_name', 'Unknown')}') is not a string.")
                                self.assertIsInstance(choice_value, (str, int, float, bool),
                                                      f"Actual value for a choice in param '{param_name}' ('{op_config.get('op_type_name', 'Unknown')}') is not a valid ParamValue type (str, int, float, bool). Got {type(choice_value)}")


if __name__ == '__main__':
    unittest.main()