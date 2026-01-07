import modal
import numpy as np
import io
from paddleocr import PaddleOCR
import cv2 as cv

image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04",
        add_python="3.10"
    )
    .pip_install(
        "paddlepaddle-gpu==2.6.2",
        "paddleocr",
        "numpy",
        "opencv-python"
    )
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libgomp1"
    )
)

app = modal.App("paddle-ocr-processing")
ocr = None

@app.cls(gpu="A10", image=image)
class OCRService:
    @modal.enter()
    def enter(self):
        global ocr

        from paddleocr import PaddleOCR
        if ocr == None:
            ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=736,
                text_det_box_thresh=0.4,
            )
            print("OCR model loaded successfully!!")

    @modal.method()
    def process_image(self, img_bytes):
        import cv2 as cv

        nparr = np.frombuffer(img_bytes, np.uint8)
        print("Recieved the images!!")
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        print("Decoded the images")

        result = ocr.predict(img)
        print("final predictions made returning the result")
        return result


@app.function(gpu="A10", image=image)
def ocr_processing_function(img_bytes):
    global ocr
    if ocr==None:
        ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=736,
            text_det_box_thresh=0.4,
            device="gpu"
        )
    print("Model loaded successfully!!")
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    result = ocr.predict(img)
    return result
