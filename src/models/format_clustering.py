import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

def cluster_results(results):
    """
    Extract structured test rows using sklearn for column clustering and classification.
    """
    # Parameters
    x_threshold = 100  # Equivalent to 'eps' in DBSCAN
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
    if len(unique_labels) > 4:
        raise ValueError(f"Too many distinct columns detected; expected at most 4. Found: {len(unique_labels)}")
    
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
    
    # Handle dynamic column counts (fallback logic)
    if len(cluster_centers) == 3:
        # Custom mapping for 3 columns: Name, Result, Units
        semantic_map = ["test_name", "result", "units"]
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