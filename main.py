import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from os.path import abspath, join as path_join
BASE_MODEL_PATH = abspath("./models")
IMAGE_PATH = abspath("./data/images")
MEDIAPIPE_MODELS_PATH = abspath("./models/mediapipe")

model_path = path_join(MEDIAPIPE_MODELS_PATH, "pose_landmarker_full.task")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

with PoseLandmarker.create_from_options(options) as landmarker:


    # Load the input image from an image file.
    mp_image = mp.Image.create_from_file(path_join(IMAGE_PATH, "ski.jpg"))

    pose_landmarker_result = landmarker.detect(mp_image)

    # Try Messing Around with the landmarks # https://github.com/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb (Try using some of that)
