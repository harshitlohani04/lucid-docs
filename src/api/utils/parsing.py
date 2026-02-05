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
