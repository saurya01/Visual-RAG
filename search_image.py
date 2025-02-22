import json
import torch
import faiss
import open_clip
import numpy as np
from PIL import Image
import os
import argparse  # For command-line arguments

# Step 1: Parse Command-Line Arguments
parser = argparse.ArgumentParser(description="Search for similar images using FAISS and CLIP.")
parser.add_argument("query_image", type=str, help="Path to the query image.")
parser.add_argument("--top_k", type=int, default=1, help="Number of top similar images to retrieve.")
args = parser.parse_args()

# Step 2: Load CLIP Model
print("[INFO] Initializing CLIP model for image retrieval...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = open_clip.create_model_and_transforms('ViT-B/32-quickgelu', pretrained='openai')
print(f"[INFO] Using device: {device}")

# Step 3: Load FAISS Index and Metadata
faiss_index_path = "image_embeddings.faiss"
metadata_path = "image_metadata.json"
coco_json_path = r"C:\Users\saury\Downloads\Oxford Pets.v1-by-breed.coco\train\_annotations.coco.json"  # Update with correct path

print("[INFO] Loading FAISS index and metadata...")
index = faiss.read_index(faiss_index_path)

with open(metadata_path, "r") as f:
    image_store = json.load(f)  # Dictionary mapping index -> (image_path, category IDs)

print(f"[INFO] FAISS index loaded. Number of stored images: {len(image_store)}")

# Step 4: Load COCO Categories
def load_coco_categories(coco_json_path):
    """Load COCO category names and create a mapping from category_id to category_name."""
    try:
        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)

        category_mapping = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        print(f"[INFO] Loaded {len(category_mapping)} categories.")
        return category_mapping
    except Exception as e:
        print(f"[ERROR] Failed to load COCO categories: {e}")
        return {}

category_mapping = load_coco_categories(coco_json_path)

# Step 5: Compute Image Embedding
def get_image_embedding(image_path):
    """Compute CLIP embedding for the input query image."""
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image).cpu().numpy()
        return embedding
    except Exception as e:
        print(f"[ERROR] Failed to process image {image_path}: {e}")
        return None

# Step 6: Search FAISS for Similar Images
def search_similar_image(query_image_path, top_k=1):
    """Search for the most similar image(s) in FAISS and return their category names."""
    print(f"[INFO] Processing query image: {query_image_path}")

    if not os.path.exists(query_image_path):
        print("[ERROR] Query image does not exist!")
        return None

    query_embedding = get_image_embedding(query_image_path)
    if query_embedding is None:
        return None

    # Perform search in FAISS
    print("[INFO] Searching FAISS index...")
    distances, indices = index.search(query_embedding, top_k)  # Find top_k nearest neighbors

    results = []
    for idx in indices[0]:
        if str(idx) in image_store:
            image_path, category_ids = image_store[str(idx)]
            
            # Convert category IDs to names
            category_names = [category_mapping.get(cat_id, "Unknown") for cat_id in category_ids]
            
            results.append({"image_path": image_path, "category_names": category_names})

    return results

# Step 7: Run the Search
query_image = args.query_image  # Get from command-line
top_k = args.top_k  # Number of top results

results = search_similar_image(query_image, top_k=top_k)

# Step 8: Display Results
if results:
    print("\n[RESULT] Most similar image(s) found:")
    for res in results:
        print(f"- Image Path: {res['image_path']}")
        print(f"  Categories: {', '.join(res['category_names'])}")
else:
    print("[INFO] No similar images found.")
