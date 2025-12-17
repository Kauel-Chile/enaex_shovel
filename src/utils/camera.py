from config.settings import FOCAL_LENGTH, SENSOR_WIDTH, SENSOR_HEIGHT
from src.domain.schemas.camera import CameraParameters

def load_camera_params():
    print(FOCAL_LENGTH, SENSOR_WIDTH, SENSOR_HEIGHT)
    return CameraParameters(
    focal_length=float(FOCAL_LENGTH),
    sensor_width=float(SENSOR_WIDTH),
    sensor_height=float(SENSOR_HEIGHT)
) 
