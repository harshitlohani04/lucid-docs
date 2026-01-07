import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import APIRouter, File, UploadFile, HTTPException, FastAPI
import cv2 as cv
from pdf2image import convert_from_bytes
from uuid import uuid4
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageOps
import easyocr
import pymupdf
from models.ner import normal_ner, custom_ner

# app = APIRouter()
app = FastAPI()

@app.post("/")
async def report_upload(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a pdf")
    else:
        report_id = str(uuid4())
    file_bytes = file.file.read()
    try:
        pages = convert_from_bytes(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not convert pdf to images : {e}")
    
    try:
        with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text = "".join(page.get_text())
                output = custom_ner(text)
                print(output)
                for name, _ in output:
                    rects = page.search_for(name)
                    for rect in rects:
                        page.add_redact_annot(rect, fill = (0, 0, 0))
                page.apply_redactions()
            doc.save(f"./local_storage/output_redacted_{report_id}.pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch text from pdf : {e}")

    return {
        "status": "success",
        "pages_processed": len(pages),
        "task_id": report_id
        }
