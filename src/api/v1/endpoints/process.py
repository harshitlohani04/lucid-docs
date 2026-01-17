from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import numpy as np
from pdf2image import convert_from_path, pdfinfo_from_path
from paddleocr import PaddleOCR
import fastapi
import modal
import io
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

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
            box_list = box.tolist()
            
            # compute axis-aligned bounding box from polygon points
            xmin, ymin = float("inf"), float("inf")
            xmax, ymax = -float("inf"), -float("inf")
            for coords in box_list:
                x, y = float(coords[0]), float(coords[1])
                xmin = min(xmin, x)
                ymin = min(ymin, y)
                xmax = max(xmax, x)
                ymax = max(ymax, y)

            # Normalize to four-corner box (top-left, top-right, bottom-right, bottom-left)
            # and convert to ints for downstream compatibility with other fixtures in the repo
            final_box_coords = [
                int(round(xmin)), int(round(xmax)),
                int(round(ymin)), int(round(ymax))
            ]

            parsed_text.append(
                {
                    "text": text,
                    "box": final_box_coords
                }
            )

    return {page_num: parsed_text[32:]}  # Skip first 32 entries which are often headers/footers


def format_ocr_results_sklearn(results):
    """
    Extract structured test rows using sklearn for column clustering and classification.
    """
    # Parameters
    x_threshold = 20  # Equivalent to 'eps' in DBSCAN
    y_threshold = 12  
    
    # 1. Flatten results -> list of items
    items = []
    for page_dict in results:
        for page_num, parsed_text in page_dict.items():
            for entry in parsed_text:
                text = (entry.get("text") or "").strip()
                box = entry.get("box")
                if not text or not box:
                    continue
                # Calculate centroids
                cx = (box[0] + box[1]) / 2
                cy = (box[2] + box[3]) / 2
                items.append({
                    "page": page_num, 
                    "text": text, 
                    "box": box, 
                    "cx": cx, 
                    "cy": cy
                })

    if not items:
        return []

    # ---------------------------------------------------------
    # SKLEARN STEP 1: Cluster X-coordinates to find columns
    # ---------------------------------------------------------
    
    # Prepare data for sklearn (requires 2D array)
    # We use a sample to determine columns, or all items if performance allows.
    # Using all items is safer to capture all column variations.
    X_coords = np.array([item["cx"] for item in items]).reshape(-1, 1)

    # DBSCAN: groups points within 'eps' distance (x_threshold)
    # min_samples=1 ensures even a single distinct header is treated as a column
    clustering = DBSCAN(eps=x_threshold, min_samples=1).fit(X_coords)
    
    # ---------------------------------------------------------
    # SKLEARN STEP 2: Map Clusters to Field Names
    # ---------------------------------------------------------
    
    # Find unique cluster labels
    unique_labels = set(clustering.labels_)
    
    # Calculate the center (mean x) of each cluster
    cluster_centers = []
    for label in unique_labels:
        if label == -1: continue # Skip noise if any
        
        # Get all x_coords belonging to this cluster
        members = X_coords[clustering.labels_ == label]
        center = members.mean()
        cluster_centers.append((center, label))
    
    # Sort clusters left-to-right based on their center X value
    cluster_centers.sort(key=lambda x: x[0])
    
    # Create the semantic mapping based on sorted order
    # 0 -> test_name, 1 -> result, etc.
    field_order = ["test_name", "result", "units", "range"]
    
    # We create a dictionary to map the distinct cluster LABEL to our semantic FIELD
    # Logic: If we found 3 clusters, map to first 3 fields, etc.
    label_to_field = {}
    
    # Handle dynamic column counts (fallback logic)
    if len(cluster_centers) == 3:
        # Custom mapping for 3 columns: Name, Result, Range (skip units)
        semantic_map = ["test_name", "result", "range"]
    elif len(cluster_centers) == 2:
        semantic_map = ["test_name", "result"]
    else:
        # Default: just take first N fields available
        semantic_map = field_order[:len(cluster_centers)]

    # ---------------------------------------------------------
    # SKLEARN STEP 3: Train Classifier (KNN) for direct lookup
    # ---------------------------------------------------------
    
    # We train a 1-Nearest Neighbor classifier on the cluster centers.
    # X = [[center1], [center2]...], y = [0, 1, 2...] (indices of semantic_map)
    
    training_centers = np.array([c[0] for c in cluster_centers]).reshape(-1, 1)
    training_labels = np.arange(len(semantic_map)) # 0, 1, 2...
    
    if len(training_centers) > 0:
        knn = KNeighborsClassifier(n_neighbors=1).fit(training_centers, training_labels)
    else:
        knn = None

    # ---------------------------------------------------------
    # Group by Rows (Y-axis) and Classify
    # ---------------------------------------------------------
    
    # Sort by Page then Y
    items_sorted = sorted(items, key=lambda x: (x["page"], x["cy"]))
    
    rows = []
    current_row = [items_sorted[0]]
    current_y = items_sorted[0]["cy"]
    
    # Simple row grouping loop (sklearn is overkill for 1D sorted gap detection)
    for it in items_sorted[1:]:
        if abs(it["cy"] - current_y) <= y_threshold:
            current_row.append(it)
            # update running mean
            current_y = sum([r["cy"] for r in current_row]) / len(current_row)
        else:
            rows.append(current_row)
            current_row = [it]
            current_y = it["cy"]
    if current_row:
        rows.append(current_row)

    structured_data = []
    
    for row_idx, row in enumerate(rows):
        row_content = {f: [] for f in field_order}
        
        for item in row:
            if knn:
                # --- THE DIRECT CLASSIFICATION ---
                # Predict which column index this item belongs to
                col_idx = knn.predict([[item["cx"]]])[0]
                field_name = semantic_map[col_idx]
                row_content[field_name].append(item["text"])
            else:
                # Fallback if clustering failed
                row_content["test_name"].append(item["text"])

        # Join text parts
        final_row = {
            "test_name": " ".join(row_content["test_name"]),
            "result": " ".join(row_content["result"]),
            "units": " ".join(row_content["units"]),
            "range": " ".join(row_content["range"]),
            "page": row[0]["page"],
            "row_index": row_idx
        }
        
        # Filter empty noise rows
        if final_row["test_name"] or final_row["result"]:
            structured_data.append(final_row)

    return structured_data


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

