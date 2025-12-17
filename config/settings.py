import os
from dotenv import load_dotenv

load_dotenv()

STORAGE_CONN_STR = os.getenv("storage_account_conn_str")
STORAGE_CONTAINER_NAME = os.getenv("storage_container_name")  

QUEUE_CONN_STR = os.getenv("service_bus_conn_str")
QUEUE_INPUT_NAME = os.getenv("queue_input_name")
QUEUE_OUTPUT_NAME = os.getenv("queue_output_name")

FOCAL_LENGTH = os.getenv("focal_length")
SENSOR_WIDTH = os.getenv("sensor_width")
SENSOR_HEIGHT = os.getenv("sensor_height")

PATH_MODEL = os.getenv("path_model")