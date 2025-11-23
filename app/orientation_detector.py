import os
import sys
import importlib.util
import torch
from PIL import Image

APP_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
DETECTION_DIR = os.path.join(ROOT_DIR, "deep-image-orientation-detection")

if not os.path.isdir(SRC_DIR):
    raise RuntimeError(f"src directory not found at: {SRC_DIR}")

if not os.path.isdir(DETECTION_DIR):
    raise RuntimeError(
        f"deep-image-orientation-detection directory not found at: {DETECTION_DIR}"
    )

# Добавляем корень проекта в sys.path, чтобы импортировать пакет src.*
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

config_path = os.path.join(SRC_DIR, "config.py")
if not os.path.isfile(config_path):
    raise RuntimeError(f"config.py not found at: {config_path}")

spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
sys.modules["config"] = config

from src.model import get_orientation_model
from src.utils import get_device, get_data_transforms, load_image_safely


class OrientationDetector:
    """ определение ориентации изображения с помощью нейросети """

    def __init__(self, model_path: str = None):
        # путь к модели по умолчанию
        if model_path is None:
            model_path = os.path.join(DETECTION_DIR, "models", "best_model.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path} "
            )

        self.model_path = model_path
        self.device = get_device()
        self.transforms = get_data_transforms()["val"]

        # загрузка модели
        self.model = get_orientation_model(pretrained=False)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # соответствие классов углам поворота
        # 0: 0°
        # 1: 90° по часовой -> -90
        # 2: 180°
        # 3: 90° против часовой -> 90
        self.angle_map = {0: 0, 1: -90, 2: 180, 3: 90}

    def predict_orientation_pil(self, image: Image.Image) -> int:
        """
        Предсказывает ориентацию по PIL.Image и возвращает угол поворота.
        """
        input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted_idx = torch.max(output, 1)

        predicted_class = predicted_idx.item()
        return self.angle_map[predicted_class]

    def predict_orientation(self, image_path: str) -> int:
        """
        Вариант интерфейса по пути к файлу (если вдруг понадобится).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = load_image_safely(image_path)
        return self.predict_orientation_pil(image)