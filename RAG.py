import json
import torch
import faiss
import open_clip
import numpy as np
from PIL import Image
import os

# Step 1: Load CLIP Model for Image Embedding Extraction
print("[INFO] Initializing CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = open_clip.create_model_and_transforms('ViT-B/32-quickgelu', pretrained='openai')
print(f"[INFO] Using device: {device}")

# Step 2: Set Up FAISS for Efficient Image Retrieval
embedding_dim = 512  # CLIP ViT-B/32 output size
index = faiss.IndexFlatL2(embedding_dim)
image_store = {}

def load_coco_annotations(coco_json_path):
    """Load COCO JSON annotations and return a mapping of image IDs to filenames and category annotations."""
    print("[INFO] Loading COCO annotations...")
    try:
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
        annotations = {img_id: [] for img_id in image_id_to_filename.keys()}
        
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            annotations[img_id].append(ann["category_id"])  # Store category IDs
        
        print(f"[INFO] Loaded {len(image_id_to_filename)} images and {len(coco_data['annotations'])} annotations.")
        return image_id_to_filename, annotations

    except Exception as e:
        print(f"[ERROR] Failed to load COCO annotations: {e}")
        return {}, {}

def get_image_embedding(image_path):
    """Compute and return the CLIP embedding for an image."""
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image).cpu().numpy()
        return embedding
    except Exception as e:
        print(f"[ERROR] Failed to process image {image_path}: {e}")
        return None

def add_images_to_faiss(image_dir, image_id_to_filename, annotations):
    """Compute embeddings and store them in FAISS."""
    print("[INFO] Computing image embeddings and storing in FAISS index...")
    num_images_processed = 0

    for img_id, filename in image_id_to_filename.items():
        image_path = os.path.join(image_dir, filename)

        if os.path.exists(image_path):
            embedding = get_image_embedding(image_path)
            if embedding is not None:
                index.add(embedding)
                image_store[len(image_store)] = (image_path, annotations[img_id])  # Store metadata
                num_images_processed += 1
        else:
            print(f"[WARNING] Image not found: {image_path}")

    print(f"[INFO] Successfully processed {num_images_processed}/{len(image_id_to_filename)} images.")

def save_faiss_index(index_path, metadata_path):
    """Save FAISS index and metadata for future retrieval."""
    print("[INFO] Saving FAISS index and metadata...")
    try:
        faiss.write_index(index, index_path)
        with open(metadata_path, "w") as f:
            json.dump(image_store, f)
        print(f"[INFO] FAISS index saved to {index_path}")
        print(f"[INFO] Metadata saved to {metadata_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save FAISS index or metadata: {e}")

# Step 3: Set File Paths
coco_json_path = r"C:\Users\saury\Downloads\Oxford Pets.v1-by-breed.coco\train\_annotations.coco.json"
image_dir = r"C:\Users\saury\Downloads\Oxford Pets.v1-by-breed.coco\train"
faiss_index_path = "image_embeddings.faiss"
metadata_path = "image_metadata.json"

# Step 4: Process Images and Store in FAISS
image_id_to_filename, annotations = load_coco_annotations(coco_json_path)
if image_id_to_filename:
    add_images_to_faiss(image_dir, image_id_to_filename, annotations)

    # Step 5: Save FAISS Index and Metadata
    save_faiss_index(faiss_index_path, metadata_path)
    print(f"[SUCCESS] FAISS index and metadata saved successfully!")
else:
    print("[ERROR] No valid COCO annotations found. Exiting.")
