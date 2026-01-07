from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import numpy as np
from pdf2image import convert_from_path, pdfinfo_from_path
from paddleocr import PaddleOCR
import fastapi
import modal
import io

ocr_engine = modal.Cls.from_name("paddle-ocr-processing", "OCRService")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_engine
    print("Loading OCR model...")
    ocr_engine = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='en',  # Good practice to specify language
        cpu_threads = 5,
        enable_mkldnn = True,
        text_det_limit_side_len=736,
        text_det_box_thresh=0.4,
    )
    print("OCR model loaded successfully.")
    yield
    
    print("Shutting down...")
    ocr_engine = None

app = FastAPI(lifespan=lifespan)

def parse_ocr(output_ocr, page_num):
    data = output_ocr
    texts = data.get("rec_texts", [])
    boxes = data.get("dt_polys", [])

    parsed_text = []

    for text, box in zip(texts, boxes):
        if text.strip():
            parsed_text.append(
                {
                    "text": text,
                    "box": box.tolist()
                }
            )

    return {page_num: parsed_text}

def format_ocr_results(results):
    pass

@app.get("/ocr-{report_id}")
async def ocr_llm_pipeline(report_id: str):
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR Model not initialized")

    print(f"Processing report: {report_id}")
    file_path = f"./local_storage/output_redacted_{report_id}.pdf"
    results = []

    try:
        info = await fastapi.concurrency.run_in_threadpool(pdfinfo_from_path, file_path)
        total_pages = info["Pages"]
        print(f"Total pages to process: {total_pages}")

        for i in range(1, total_pages + 1):

            current_page_list = await fastapi.concurrency.run_in_threadpool(
                convert_from_path, 
                file_path, 
                first_page=i, 
                last_page=i,
                dpi=150 # Keeping DPI reasonable for speed
            )
            
            if not current_page_list:
                continue
                
            page_image = current_page_list[0]

            # img_byte_arr = io.BytesIO()
            # page_image.save(img_byte_arr, format="JPEG")
            # img_bytes = img_byte_arr.getvalue()
            img_array = np.array(page_image)

            try:
                # ocr = modal.Function.from_name("paddle-ocr-processing", "ocr_processing_function")
                ocr_result = await fastapi.concurrency.run_in_threadpool(ocr_engine.predict, img_array)
                if ocr_result and ocr_result[0]:
                    parsed_data = await fastapi.concurrency.run_in_threadpool(
                        parse_ocr, 
                        ocr_result[0], 
                        i
                    )
                    results.append(parsed_data)
                del page_image
                del img_array
            except Exception as e:
                raise HTTPException(status_code="400", detail=f"Modal error : {e}")
            
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return {
        "status": "paddle ocr success",
        "task_id": report_id,
        "extracted_text": results
    }

