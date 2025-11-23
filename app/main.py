from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
import os
import numpy as np
import cv2
import uvicorn

from app.detector import SignatureDetector
from app.classificator import DocumentClassificator
from app.preprocessing import Preprocessor

# корневая папка проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# пути к моделям по умолчанию, можно переопределить через переменные окружения
SIGNATURE_MODEL_PATH = os.getenv(
    "SIGNATURE_MODEL_PATH",
    os.path.join(MODELS_DIR, "signature-AD.pt"),
)
CLASSIFICATOR_MODEL_PATH = os.getenv(
    "CLASSIFICATOR_MODEL_PATH",
    os.path.join(MODELS_DIR, "classificator-best.pt"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Проверяем, что модели существуют
    if not os.path.exists(SIGNATURE_MODEL_PATH):
        raise Exception(f"Signature model not found: {SIGNATURE_MODEL_PATH}")
    if not os.path.exists(CLASSIFICATOR_MODEL_PATH):
        raise Exception(f"Classifier model not found: {CLASSIFICATOR_MODEL_PATH}")

    # Инициализируем классы один раз
    app.state.preprocessor = Preprocessor()
    app.state.classificator = DocumentClassificator(CLASSIFICATOR_MODEL_PATH)
    app.state.detector = SignatureDetector(SIGNATURE_MODEL_PATH)

    yield


app = FastAPI(
    title="Signature Detection API",
    lifespan=lifespan
)


@app.post("/detect-signatures")
async def detect_signatures(file: UploadFile = File(...)):
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Allowed: jpg, jpeg, png",
        )

    try:
        # --- 0. Загрузка изображения ---
        content = await file.read()
        file_bytes = np.frombuffer(content, np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Cannot open image file")

        #тут бинаризация для детекции

        # # --- 1. ПРЕДОБРАБОТКА (ориентация + бинаризация) ---
        # preprocessor: Preprocessor = app.state.preprocessor
        # rotated_bgr, binarized = preprocessor.process(image_bgr)
        #
        # # --- 2. КЛАССИФИКАЦИЯ (используем rotated_bgr) ---
        # classificator: DocumentClassificator = app.state.classificator
        # doc_type = classificator.classify_document(rotated_bgr)
        #
        # if doc_type == "handwritten":
        #     return {
        #         "document_type": doc_type,
        #         "message": "Handwritten documents are not processed",
        #     }
        #
        # # детекция подписей на binarized
        # # YOLO требует 3 канала — конвертируем GRAY → BGR
        # binarized_3ch = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
        #
        # detector: SignatureDetector = app.state.detector
        # signature_count = detector.count_signatures(binarized_3ch)
        #
        # return {
        #     "document_type": doc_type,
        #     "number_of_signatures": signature_count,
        # }

        #тут бинаризация для классификации

        # # 1. ПРЕДОБРАБОТКА: ориентация + бинаризация
        # preprocessor: Preprocessor = app.state.preprocessor
        # rotated_bgr, binarized = preprocessor.process(image_bgr)
        #
        # # 2. КЛАССИФИКАЦИЯ ДОКУМЕНТА
        # classificator: DocumentClassificator = app.state.classificator
        # doc_type = classificator.classify_document(binarized)
        #
        # if doc_type == "handwritten":
        #     return {
        #         "document_type": doc_type,
        #         "message": "Handwritten documents are not processed",
        #     }
        #
        # # 3. ДЕТЕКЦИЯ ПОДПИСЕЙ (только для печатных)
        # detector: SignatureDetector = app.state.detector
        # signature_count = detector.count_signatures(rotated_bgr)
        #
        # return {
        #     "document_type": doc_type,
        #     "number_of_signatures": signature_count,
        # }

        # тут все BGR

            # 1. предобработка: только нормализация ориентации, работаем в BGR
        preprocessor: Preprocessor = app.state.preprocessor
        rotated_bgr = preprocessor.correct_orientation(image_bgr)

            # 2. классификация документа по BGR-изображению
        classificator: DocumentClassificator = app.state.classificator
        doc_type = classificator.classify_document(rotated_bgr)

        if doc_type == "handwritten":
            return {
                "document_type": doc_type,
                "message": "Handwritten documents are not processed",
            }

            # 3. детекция подписей также по BGR-изображению
        detector: SignatureDetector = app.state.detector
        signature_count = detector.count_signatures(rotated_bgr)

        return {
            "document_type": doc_type,
            "number_of_signatures": signature_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)