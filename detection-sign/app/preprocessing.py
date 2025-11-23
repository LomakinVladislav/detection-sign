import cv2
import numpy as np
from PIL import Image

from app.orientation_detector import OrientationDetector


class Preprocessor:

    def __init__(self):
        self.orientation_detector = OrientationDetector()

    def image_binarized(self, image_bgr: np.ndarray) -> np.ndarray:
        """ бинаризация изображения """
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr

        max_output_value = 255
        neighborhood_size = 99
        substract_from_mean = 10

        image_binarized = cv2.adaptiveThreshold(
            gray,
            max_output_value,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            neighborhood_size,
            substract_from_mean,
        )

        return image_binarized

    def correct_orientation(self, image_bgr: np.ndarray) -> np.ndarray:
        """ исправляет ориентацию изображения """
        # BGR -> RGB -> PIL
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        angle = self.orientation_detector.predict_orientation_pil(pil_image)

        if angle != 0:
            # PIL.rotate: положительный угол — против часовой, поэтому
            # -90 (класс 1) = поворот на 90° по часовой
            pil_image = pil_image.rotate(angle, expand=True)

        # обратно в BGR
        rotated_rgb = np.array(pil_image)
        rotated_bgr = cv2.cvtColor(rotated_rgb, cv2.COLOR_RGB2BGR)
        return rotated_bgr

    def process(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        бинаризация + ориентация, возвращает нормализированное и бинаризированное изображение
        """
        rotated = self.correct_orientation(image_bgr)
        binarized = self.image_binarized(rotated)
        return rotated, binarized

# if __name__ == "__main__":
#     # ---- ВСТАВЬ СЮДА СВОЙ ПУТЬ К ИЗОБРАЖЕНИЮ ----
#     image_path = "/Users/an.kornn/PycharmProjects/Kazna/tests/data/тест_изобр.jpg"   # ← поменяй на свой путь
#     # --------------------------------------------
#
#     import os
#
#     # читаем BGR изображение
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Cannot read image: {image_path}")
#
#     pre = Preprocessor()
#
#     # получаем два изображения
#     rotated, binarized = pre.process(img)
#
#     # формируем пути для сохранения
#     base, ext = os.path.splitext(image_path)
#     rotated_path = f"{base}_rotated.jpg"
#     binarized_path = f"{base}_binarized.jpg"
#
#     # сохраняем
#     cv2.imwrite(rotated_path, rotated)
#     cv2.imwrite(binarized_path, binarized)
#
#     print("Saved rotated image to:", rotated_path)
#     print("Saved binarized image to:", binarized_path)
