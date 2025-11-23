from ultralytics import YOLO
import numpy as np


class DocumentClassificator:
    def __init__(self, model_path: str = "models/classificator.pt"):
        self.model = YOLO(model_path)
        self.class_names = {0: "handwritten", 1: "printed"}

    def classify_document(self, binarized_image: np.ndarray) -> str:
        """ классифицирует документ и возвращает его тип """
        results = self.model(binarized_image, verbose=False)

        for r in results:
            if r.probs is not None:
                class_id = r.probs.top1
                return self.class_names.get(int(class_id), "uknown")

        return "uknown"

