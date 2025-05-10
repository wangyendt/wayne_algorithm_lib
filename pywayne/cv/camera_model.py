# author: wangye(Wayne)
# license: Apache Licence
# file: camera_model.py
# time: 2024-10-26-16:05:00  # Adjusted timestamp
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import os
import sys
import subprocess
import importlib
import numpy as np
from typing import Union, Tuple, Dict, Any, List
from pathlib import Path
from pywayne.cv.tools import write_cv_yaml # Use write_cv_yaml from tools

class CameraModel:
    """
    A wrapper class for the camera_models C++ library exposed via pybind11.
    Handles library checking, loading models from YAML, and provides access
    to common camera operations and parameters.
    """
    def __init__(self):
        """
        Initializes the CameraModel, checking for the required library
        and setting up the camera factory.
        """
        self.camera_models_module = self._check_and_import_lib()
        if self.camera_models_module:
            self.camera_factory = self.camera_models_module.CameraFactory.instance()
        else:
            self.camera_factory = None # Indicate failure
        self.camera = None # Holds the loaded camera object (pybind wrapper)
        self.parameters = None # Holds the parameters object (pybind wrapper)

    def _check_and_import_lib(self):
        """
        Checks if the 'camera_models' pybind module is available.
        If not, attempts to download/build it using 'gettool'.
        Returns the imported module or None if unsuccessful.
        """
        lib_name = "camera_models"
        tool_name = "camera_models" # Tool name might be different, adjust if needed
        lib_path = Path(os.path.dirname(os.path.abspath(__file__))) / 'lib'

        # Ensure lib path is in sys.path for import
        if str(lib_path) not in sys.path:
             sys.path.append(str(lib_path))

        try:
            # Try importing first
            return importlib.import_module(lib_name)
        except ImportError:
            print(f"'{lib_name}' module not found. Attempting to acquire using 'gettool'...")
            try:
                os.makedirs(lib_path, exist_ok=True)
                # Ensure gettool is available in PATH or provide full path if necessary
                subprocess.run(['gettool', tool_name, '-b', '-t', str(lib_path)], check=True)
                print(f"Successfully acquired '{tool_name}' into {lib_path}")
                importlib.invalidate_caches() # Ensure import system sees the new module
                return importlib.import_module(lib_name)
            except FileNotFoundError:
                 print(f"Error: 'gettool' command not found. Please ensure it's installed and in your PATH.")
                 return None
            except subprocess.CalledProcessError as e:
                print(f"Error running 'gettool {tool_name}':")
                print(f"Command: {' '.join(e.cmd)}")
                print(f"Return Code: {e.returncode}")
                print(f"Output: {e.output}")
                print(f"Stderr: {e.stderr}")
                print(f"Failed to acquire '{tool_name}'. Please check the 'gettool' setup and try again.")
                return None
            except ImportError:
                 print(f"Error: Could not import '{lib_name}' even after running 'gettool'. Check installation.")
                 return None
            except Exception as e:
                print(f"An unexpected error occurred during library acquisition: {e}")
                return None

    def load_from_yaml(self, yaml_path: Union[str, Path]):
        """
        Loads a camera model from the specified YAML configuration file.

        Args:
            yaml_path: Path to the camera configuration YAML file.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            RuntimeError: If the camera factory is not available or loading fails.
        """
        if not self.camera_factory:
             raise RuntimeError("CameraFactory not initialized. Check library installation.")

        yaml_file = Path(yaml_path)
        if not yaml_file.is_file():
            raise FileNotFoundError(f"Camera configuration file not found: {yaml_path}")

        try:
            self.camera = self.camera_factory.generate_camera_from_yaml_file(str(yaml_file))
            # Attempt to get the specific parameters object
            if hasattr(self.camera, "get_parameters"):
                # Assuming get_parameters returns the correct derived Parameter type
                self.parameters = self.camera.get_parameters()
            else:
                # Fallback: maybe try accessing a base parameter object if available
                # Or simply leave self.parameters as None if no specific getter exists
                 self.parameters = None
                 print("Warning: Could not retrieve specific parameters object. Parameter details might be limited.")

            print(f"Successfully loaded camera '{self.camera_name}' (Type: {self.model_type}) from {yaml_file}")
        except Exception as e:
            # Catch pybind exceptions or other errors during loading
            self.camera = None
            self.parameters = None
            raise RuntimeError(f"Failed to load camera model from {yaml_path}: {e}") from e

    def _ensure_camera_loaded(self):
        """Raises RuntimeError if the camera model hasn't been loaded."""
        if self.camera is None:
            raise RuntimeError("Camera model not loaded. Call load_from_yaml() first.")

    # --- Properties ---
    @property
    def model_type(self) -> Any: # Returns the pybind enum type
        """The model type of the loaded camera (e.g., PINHOLE, MEI)."""
        self._ensure_camera_loaded()
        return self.camera.model_type

    @property
    def camera_name(self) -> str:
        """The name of the loaded camera."""
        self._ensure_camera_loaded()
        return self.camera.camera_name

    @property
    def image_width(self) -> int:
        """The width of the camera image in pixels."""
        self._ensure_camera_loaded()
        return self.camera.image_width

    @property
    def image_height(self) -> int:
        """The height of the camera image in pixels."""
        self._ensure_camera_loaded()
        return self.camera.image_height

    # --- Core Camera Methods ---
    def lift_projective(self, p: Union[Tuple[float, float], List[float], np.ndarray]) -> np.ndarray:
        """
        Lifts a 2D image point to a 3D projective ray (unit vector).

        Args:
            p: A tuple, list, or NumPy array (u, v) representing the image point coordinates.

        Returns:
            A NumPy array (x, y, z) representing the 3D direction vector.
        """
        self._ensure_camera_loaded()
        try:
            # The pybind layer should handle tuple/list/ndarray -> Eigen::Vector2d
            # and Eigen::Vector3d -> numpy array.
            P = self.camera.lift_projective(p)
            return np.array(P, dtype=np.float64) # Ensure output is np.ndarray
        except Exception as e:
            raise RuntimeError(f"Error during lift_projective for point {p}: {e}") from e

    def space_to_plane(self, P: Union[Tuple[float, float, float], List[float], np.ndarray]) -> np.ndarray:
        """
        Projects a 3D point in camera coordinates onto the 2D image plane.

        Args:
            P: A tuple, list, or NumPy array (x, y, z) representing the 3D point.

        Returns:
            A NumPy array (u, v) representing the projected image point coordinates.
        """
        self._ensure_camera_loaded()
        try:
            # pybind handles tuple/list/ndarray -> Eigen::Vector3d and Eigen::Vector2d -> numpy array
            p_out = self.camera.space_to_plane(P) # Renamed to avoid conflict with input p for lift_projective
            return np.array(p_out, dtype=np.float64) # Ensure output is np.ndarray
        except Exception as e:
            raise RuntimeError(f"Error during space_to_plane for point {P}: {e}") from e

    def get_parameters_as_dict(self) -> Dict[str, Any]:
        """
        Returns the loaded camera parameters as a dictionary. Includes base parameters
        and attempts to include model-specific parameters if available.

        Returns:
            A dictionary containing the camera parameters.
        """
        self._ensure_camera_loaded()

        if self.parameters is None:
            # Try getting base parameters directly from camera if specific obj failed
            base_params = {
                "model_type": str(self.camera.model_type),
                "camera_name": self.camera.camera_name,
                "image_width": self.camera.image_width,
                "image_height": self.camera.image_height,
                "warning": "Specific parameter object not available; only base info retrieved."
            }
            # Maybe add nIntrinsics if available on base camera? Check pybind defs.
            # if hasattr(self.camera, 'n_intrinsics'):
            #     base_params["n_intrinsics"] = self.camera.n_intrinsics
            return base_params


        params_dict: Dict[str, Any] = {}
        try:
            # Base parameters from the Parameter object
            params_dict["model_type"] = str(self.parameters.model_type) # Enum to string
            params_dict["camera_name"] = self.parameters.camera_name
            params_dict["image_width"] = self.parameters.image_width
            params_dict["image_height"] = self.parameters.image_height
            params_dict["n_intrinsics"] = self.parameters.n_intrinsics

            # Model-specific parameters
            param_type = type(self.parameters)
            module = self.camera_models_module

            if param_type == module.PinholeCameraParameters:
                params_dict.update({
                    "k1": self.parameters.k1, "k2": self.parameters.k2,
                    "p1": self.parameters.p1, "p2": self.parameters.p2,
                    "fx": self.parameters.fx, "fy": self.parameters.fy,
                    "cx": self.parameters.cx, "cy": self.parameters.cy,
                })
            elif param_type == module.PinholeFullCameraParameters:
                 params_dict.update({
                    "k1": self.parameters.k1, "k2": self.parameters.k2, "k3": self.parameters.k3,
                    "k4": self.parameters.k4, "k5": self.parameters.k5, "k6": self.parameters.k6,
                    "p1": self.parameters.p1, "p2": self.parameters.p2,
                    "fx": self.parameters.fx, "fy": self.parameters.fy,
                    "cx": self.parameters.cx, "cy": self.parameters.cy,
                })
            elif param_type == module.CataCameraParameters:
                 params_dict.update({
                    "xi": self.parameters.xi,
                    "k1": self.parameters.k1, "k2": self.parameters.k2,
                    "p1": self.parameters.p1, "p2": self.parameters.p2,
                    "gamma1": self.parameters.gamma1, "gamma2": self.parameters.gamma2,
                    "u0": self.parameters.u0, "v0": self.parameters.v0,
                })
            elif param_type == module.EquidistantCameraParameters:
                params_dict.update({
                    "k2": self.parameters.k2, "k3": self.parameters.k3,
                    "k4": self.parameters.k4, "k5": self.parameters.k5,
                    "mu": self.parameters.mu, "mv": self.parameters.mv,
                    "u0": self.parameters.u0, "v0": self.parameters.v0,
                })
            elif param_type == module.OCAMCameraParameters:
                 params_dict.update({
                     "C": self.parameters.C, "D": self.parameters.D, "E": self.parameters.E,
                     "center_x": self.parameters.center_x, "center_y": self.parameters.center_y,
                     "poly": list(self.parameters.poly), # Ensure it's a standard list
                     "inv_poly": list(self.parameters.inv_poly), # Ensure it's a standard list
                })
            else:
                 params_dict["info"] = "Parameter type not specifically handled for detailed dictionary output."

        except AttributeError as e:
            params_dict["error"] = f"Could not access all parameter attributes: {e}"
        except Exception as e:
             params_dict["error"] = f"An unexpected error occurred while getting parameters: {e}"

        return params_dict


# --- Main Execution Block ---
if __name__ == '__main__':
    # Define content for a sample Pinhole camera YAML file as a dictionary
    sample_yaml_data = {
        "model_type": "PINHOLE",
        "camera_name": "sample_pinhole_cam",
        "image_width": 640,
        "image_height": 480,
        "distortion_parameters": {
           "k1": -0.28,
           "k2": 0.07,
           "p1": 0.0002,
           "p2": 0.00002
        },
        "projection_parameters": {
           "fx": 460.0,
           "fy": 458.0,
           "cx": 330.0,
           "cy": 245.0
        }
    }

    # Define the filename for the sample YAML
    yaml_filename = "sample_pinhole_cam.yaml"
    temp_file_created = False

    try:
        # Create the sample YAML file using write_cv_yaml
        if not os.path.exists(yaml_filename):
            success = write_cv_yaml(yaml_filename, sample_yaml_data)
            if success:
                print(f"Created temporary sample configuration file using write_cv_yaml: {yaml_filename}")
                temp_file_created = True
            else:
                print(f"Failed to create temporary sample configuration file using write_cv_yaml: {yaml_filename}")
                # Decide how to proceed if writing fails, maybe exit?
                sys.exit(1) # Exit if sample file creation failed
        else:
            print(f"Using existing configuration file: {yaml_filename}")


        # 1. Create CameraModel instance (this checks/gets the library)
        print("\n--- Initializing CameraModel ---")
        camera_model = CameraModel()

        # Check if initialization was successful
        if not camera_model.camera_factory:
             print("Exiting example due to library initialization failure.")
             sys.exit(1) # Exit if library failed

        # 2. Load camera from YAML
        print(f"\n--- Loading Camera from {yaml_filename} ---")
        camera_model.load_from_yaml(yaml_filename)

        # 3. Print basic info using properties
        print("\n--- Camera Properties ---")
        print(f"Camera Name: {camera_model.camera_name}")
        print(f"Model Type: {camera_model.model_type}")
        print(f"Image Size: {camera_model.image_width}x{camera_model.image_height}")

        # 4. Get and print detailed parameters
        print("\n--- Detailed Parameters ---")
        params = camera_model.get_parameters_as_dict()
        if "error" in params:
             print(f"Error retrieving parameters: {params['error']}")
        for key, value in params.items():
             # Pretty print lists/long values
            if isinstance(value, list) and len(value) > 5:
                 print(f"  {key}: [{value[0]:.4f}, {value[1]:.4f}, ..., {value[-1]:.4f}] (length {len(value)})")
            elif isinstance(value, float):
                 print(f"  {key}: {value:.6f}")
            else:
                 print(f"  {key}: {value}")


        # 5. Test projection/lifting
        print("\n--- Projection/Lifting Tests ---")
        # Example 2D point (offset from center)
        cx = params.get('cx', camera_model.image_width / 2)
        cy = params.get('cy', camera_model.image_height / 2)
        image_point = (cx + 50.5, cy - 30.2)
        print(f"Input image point: {image_point}")

        # Lift point to 3D ray
        projective_ray = camera_model.lift_projective(image_point)
        print(f"Lifted projective ray (X,Y,Z): {tuple(f'{v:.6f}' for v in projective_ray)}")

        # Example 3D point in front of the camera
        world_point = (0.2, -0.1, 2.5) # (X, Y, Z) in meters
        print(f"\nInput world point: {world_point}")

        # Project 3D point back to 2D
        reprojected_point = camera_model.space_to_plane(world_point)
        print(f"Projected image point (u,v): {tuple(f'{v:.2f}' for v in reprojected_point)}")

        # Compare reprojected point to original (if world point was derived from it)
        # Note: For this example, world_point is arbitrary, so reprojection won't match image_point
        print("\n(Note: Reprojected point is for the arbitrary world point, not derived from the initial image point)")


    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the YAML file exists or can be created.")
    except RuntimeError as e:
        print(f"\nRuntime Error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # Clean up the temporary sample file if we created it
        if temp_file_created and os.path.exists(yaml_filename):
            try:
                os.remove(yaml_filename)
                print(f"\nCleaned up temporary sample file: {yaml_filename}")
            except OSError as e:
                print(f"Error removing temporary file {yaml_filename}: {e}") 
