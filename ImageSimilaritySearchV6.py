import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import cv2
from timm import create_model
from transformers import CLIPProcessor, CLIPModel
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

# Constants
BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)
SWIN_MODEL_NAME = 'swin_base_patch4_window7_224'
CLIP_VIT_MODEL_NAME = 'openai/clip-vit-base-patch32'
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# Image display constants
IMAGES_PER_ROW = 4
IMAGE_DISPLAY_SIZE = (180, 180)

# Global variables
features_file = None
current_cluster_index = 0
current_plot_window = None
current_scrollable_frame = None
current_image_frame = None
clusters = None
model = None
transform = None
model_type = None
clip_processor = None
isTextSearch = False

# GUI widget references
dataset_folder_entry = None
text_search_entry = None
input_image_entry = None
threshold_entry = None
top_k_entry = None
max_num_clusters_entry = None
min_images_per_cluster_entry = None
max_images_per_cluster_entry = None
root = None

#-------------------------- Setup Logic -----------------------------#

# Load a pre-trained models and set to evaluation mode
def setup_model_and_transform(model_type):
    global model, transform, clip_processor

    # Swin Model Setup
    if model_type == 'swin':
        print("Setting up Swin Model...")
        model = create_model(SWIN_MODEL_NAME, pretrained=True)
        model.head = torch.nn.Identity()
        
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])
    
    # CLIP Model Setup    
    else:
        print("Setting up CLIP Model...")
        model = CLIPModel.from_pretrained(CLIP_VIT_MODEL_NAME, use_safetensors=True, cache_dir=r'D:\AI Assets\CLIP')
        clip_processor = CLIPProcessor.from_pretrained(CLIP_VIT_MODEL_NAME, use_safetensors=True, cache_dir=r'D:\AI Assets\CLIP')
        transform = lambda img: clip_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

    model.eval()
    model = model.float()
    
    return model, transform

#-------------------------- Feature Extraction Logic -----------------------------#

# Extract features from a batch of images
def extract_features_batch(image_paths, model, transform, model_type):
    batch = [transform(Image.open(img_path).convert('RGB')) for img_path in image_paths]
    batch = torch.stack(batch)

    with torch.no_grad():
        features = model(batch) if model_type == 'swin' else model.get_image_features(batch)
        features = F.normalize(features, p=2, dim=1)
        features = features.cpu().numpy()
    return features

# Extract features in parallel
def extract_features_parallel(image_paths, model, transform, model_type, batch_size=BATCH_SIZE):
    all_features = []
    
    with ThreadPoolExecutor() as executor:
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i + batch_size]
            batch_features = executor.submit(extract_features_batch, batch_paths, model, transform, model_type).result()
            all_features.extend(batch_features)
    return all_features

# Save features
def save_features(features, filenames, features_file):
    with open(features_file, 'wb') as f:
        pickle.dump((features, filenames), f)
    return features_file

# Load features
def load_features(features_file, model_type):
    with open(features_file, 'rb') as f:
        features_file = pickle.load(f)
    return features_file 

# Update features and filenames in the pickle file if new images are added
def update_features(dataset_folder, features_file, model, transform, model_type):
    image_list = os.listdir(dataset_folder)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.jfif')
    dataset_filenames = [os.path.join(dataset_folder, img) for img in image_list if img.lower().endswith(valid_extensions)]

    if os.path.exists(features_file):
        existing_features, existing_filenames = load_features(features_file, model_type)
        new_image_paths = [img for img in dataset_filenames if img not in existing_filenames]

        if new_image_paths:
            print(f"Found {len(new_image_paths)} new images. Updating features...")
            new_features = extract_features_parallel(new_image_paths, model, transform, model_type)
            updated_features = existing_features + new_features
            updated_filenames = existing_filenames + new_image_paths
            
            save_features(updated_features, updated_filenames, features_file)
            return updated_features, updated_filenames
        
        return existing_features, existing_filenames
    else:
        print(f"Extracting features for {len(dataset_filenames)} images...")
        features = extract_features_parallel(dataset_filenames, model, transform, model_type)
        save_features(features, dataset_filenames, features_file)
        return features, dataset_filenames

#-------------------------- Similarity Calculations Logic -----------------------------#

# Find similar images for either a text prompt or an input image
def find_similar_images(query, dataset_folder, features_file, model, transform, top_k, similarity_threshold, model_type):
    # Get query features
    if isTextSearch and model_type == 'clip':            
        # Process text query using CLIP
        inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            query_features = model.get_text_features(**inputs)

    else:
        # Process image query
        image_to_transform = Image.open(query).convert('RGB')
        input_tensor = transform(image_to_transform).unsqueeze(0)
        
        with torch.no_grad():
            query_features = model(input_tensor) if model_type == 'swin' else model.get_image_features(input_tensor)        
    
    # L2 normalization, convert to numpy and reshape
    query_features = F.normalize(query_features, p=2, dim=1)
    query_features = query_features.cpu().numpy().reshape(1, -1)
    
    # Load dataset features
    dataset_features, filenames = update_features(dataset_folder, features_file, model, transform, model_type)
    dataset_features = np.array(dataset_features).reshape(len(dataset_features), -1)
    
    # Calculate similarities
    similarities = cosine_similarity(query_features, dataset_features)[0]
    
    # Filter by threshold and get top-k results
    filtered_indices = [i for i, sim in enumerate(similarities) if sim >= similarity_threshold]
    if not filtered_indices:
        return []
    
    # Sort and return top-k
    sorted_indices = np.argsort(similarities[filtered_indices])[-top_k:][::-1]
    return [(filenames[filtered_indices[idx]], similarities[filtered_indices[idx]]) for idx in sorted_indices]


#-------------------------- Clusters Calculations Logic -----------------------------#

# Cluster similar images
def cluster_similar_images(dataset_folder, features_file, model, transform, max_num_clusters, similarity_threshold, min_images_per_cluster, max_images_per_cluster, model_type):
    
    features, image_paths = update_features(dataset_folder, features_file, model, transform, model_type)
    features_reshaped = np.array(features).reshape(len(features), -1)
    
    similarity_matrix = cosine_similarity(features_reshaped)
    distance_matrix = 1 - similarity_matrix

    if max_num_clusters is not None and similarity_threshold is None:
        clustering = AgglomerativeClustering(n_clusters=max_num_clusters, metric='precomputed', linkage='average')

    elif max_num_clusters is None and similarity_threshold is not None:    
        clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=1 - similarity_threshold)
        
    labels = clustering.fit_predict(distance_matrix)

    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(image_paths[i])
        
    if min_images_per_cluster is not None and max_images_per_cluster is None:
        clusters = {key: value for key, value in clusters.items() if len(value) >= min_images_per_cluster}
    
    elif max_images_per_cluster is not None and min_images_per_cluster is None:
        clusters = {key: value for key, value in clusters.items() if len(value) <= max_images_per_cluster}      
        
    return clusters

#-------------------------- Plotting Logic -----------------------------#

# Fast image loading with OpenCV (aspect ratio preserved)
def load_image_fast(image_path, size=IMAGE_DISPLAY_SIZE, padding_color=(40, 40, 40)):
    try:
        # Load image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Get dimensions
        h, w = img.shape[:2]
        target_w, target_h = size
        
        # Calculate the scale factor to fit the image within target size
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions (maintaining aspect ratio)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image with preserved aspect ratio
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create a canvas with padding color
        canvas = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
        
        # Calculate centering offsets
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place the resized image on the canvas
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        # Convert BGR to RGB for PIL
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image, then CTkImage
        pil_img = Image.fromarray(img_rgb)
        return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=size)
        
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Create scrollable plot window
def create_scrollable_plot_window(title="Image Results"):
    window = ctk.CTkToplevel()
    window.title(title)
    window.geometry("900x700")
    
    # Create main frame
    main_frame = ctk.CTkFrame(window)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Create scrollable frame
    scrollable_frame = ctk.CTkScrollableFrame(main_frame, width=950, height=700)
    scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    return window, scrollable_frame

# Create fast image grid using OpenCV + CustomTkinter
def create_fast_image_grid(scrollable_frame, images_data, title, is_similarity=False):
    global current_image_frame
    
    # Clear existing image frame if it exists
    if current_image_frame:
        current_image_frame.destroy()
    
    # Create new image frame with CustomTkinter
    current_image_frame = ctk.CTkFrame(scrollable_frame)
    current_image_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Title label with CustomTkinter
    title_label = ctk.CTkLabel(
        current_image_frame, 
        text=title, 
        font=ctk.CTkFont(size=16, weight="bold")
    )
    title_label.grid(row=0, column=0, columnspan=IMAGES_PER_ROW, pady=(0, 20))
    
    if not images_data:
        no_images_label = ctk.CTkLabel(
            current_image_frame,
            text="No images to display",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        no_images_label.grid(row=1, column=0, columnspan=IMAGES_PER_ROW, pady=20)
        return
    
    # Load and display images in grid
    loaded_count = 0
    for i, data in enumerate(images_data):
        if is_similarity:
            img_path, similarity = data
            title_text = f'Similarity: {similarity*100:.1f}%'
        else:
            img_path = data
            filename = os.path.basename(img_path)
            title_text = filename[:25] + "..." if len(filename) > 25 else filename
        
        # Load image
        photo = load_image_fast(img_path)
        if photo is None:
            continue
        
        # Calculate grid position
        row = (loaded_count // IMAGES_PER_ROW) + 1
        col = loaded_count % IMAGES_PER_ROW
        
        # Create frame for image and label
        img_container = ctk.CTkFrame(
            current_image_frame, 
            border_width=0, 
            corner_radius=0,
            fg_color="transparent"
        )
        img_container.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
        
        # Image label
        img_label = ctk.CTkLabel(img_container, image=photo, text="")
        img_label.pack()
        
        # Title label
        text_label = ctk.CTkLabel(
            img_container, 
            text=title_text, 
            font=ctk.CTkFont(size=9),
            text_color='white',
            wraplength=IMAGE_DISPLAY_SIZE[0],
            justify='center'
        )
        text_label.pack(pady=(2, 0))
        
        loaded_count += 1
    
    # Configure grid weights for responsive layout
    for i in range(IMAGES_PER_ROW):
        current_image_frame.grid_columnconfigure(i, weight=1)

# Plot similar images in a scrollable window
def plot_similar_images_scrollable(similar_images):
    if not similar_images:
        return
    
    global current_plot_window, current_scrollable_frame
    
    # Close previous window if it exists
    if current_plot_window and current_plot_window.winfo_exists():
        current_plot_window.destroy()
    
    # Create new scrollable plot window
    current_plot_window, current_scrollable_frame = create_scrollable_plot_window("Similar Images")
    create_fast_image_grid(current_scrollable_frame, similar_images, f"Similar Images to Query ({len(similar_images)} found)", is_similarity=True)

# Plot image clusters in a scrollable window
def plot_image_clusters_scrollable(clusters, cluster_index=0):
    if not clusters:
        return
    
    global current_plot_window, current_scrollable_frame
    
    # Close previous window if it exists
    if current_plot_window and current_plot_window.winfo_exists():
        current_plot_window.destroy()
    
    cluster_label = list(clusters.keys())[cluster_index]
    cluster_images = clusters[cluster_label]
    
    title = f"Cluster {cluster_index + 1}/{len(clusters)} ({len(cluster_images)} images)"
    
    # Create new scrollable plot window
    current_plot_window, current_scrollable_frame = create_scrollable_plot_window("Image Clusters")
    create_fast_image_grid(current_scrollable_frame, cluster_images, title, is_similarity=False)

# Update cluster plot within existing window (for navigation)
def update_cluster_plot(clusters, cluster_index=0):
    if not clusters or not current_scrollable_frame:
        return
    
    cluster_label = list(clusters.keys())[cluster_index]
    cluster_images = clusters[cluster_label]
    
    title = f"Cluster {cluster_index + 1}/{len(clusters)} ({len(cluster_images)} images)"
    
    # Update plot within existing window
    create_fast_image_grid(current_scrollable_frame, cluster_images, title, is_similarity=False)

#-------------------------- GUI Callback Functions -----------------------------#

# Get model type
def get_model_type(choice):
    global model_type
    model_type = choice

# Get dataset folder and update features_file
def get_dataset_folder():
    global features_file, model_type, dataset_folder_entry
    dataset_folder = filedialog.askdirectory()
    
    if dataset_folder:
        folder_name = os.path.basename(os.path.normpath(dataset_folder))
        # Set the features file name based on the dataset folder's name
        features_file = f'{folder_name}_features_{model_type}.pkl'
        dataset_folder_entry.delete(0, ctk.END)
        dataset_folder_entry.insert(0, dataset_folder)
        
        print(f"Features file set to: {features_file}")

# Get input image
def get_image_path():
    global input_image_entry
    input_image = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.jfif")])
    if input_image:
        input_image_entry.delete(0, ctk.END)
        input_image_entry.insert(0, input_image)

# Run text search in thread
def run_text_search():
    global model_type, features_file, isTextSearch
    
    # Get model, dataset folder and input image
    current_model_type = model_type
    print(f"Current model type: {current_model_type}")
    dataset_folder_path = dataset_folder_entry.get().strip()
    prompt = text_search_entry.get().strip()
    isTextSearch = True
    
    # Get similarity threshold and number of similar images
    similarity_threshold = float(threshold_entry.get()) if threshold_entry.get().strip() else 0.12
    num_similar_images = int(top_k_entry.get()) if top_k_entry.get().isdigit() else 5
    
    # Get model and run text search
    model, transform = setup_model_and_transform(current_model_type)
    
    similar_images = find_similar_images(prompt, dataset_folder_path, features_file, model, transform, num_similar_images, similarity_threshold, current_model_type)
    
    # Update GUI with results
    update_gui_with_similar_images(similar_images)

# Update GUI after text search
def update_gui_with_similar_images(similar_images):
    global root
    if similar_images:
        root.after(0, lambda: plot_similar_images_scrollable(similar_images))
    else:
        root.after(0, lambda: messagebox.showinfo("Info", "No similar images found."))

# Text search function
def search_by_text():
    threading.Thread(target=run_text_search, daemon=True).start()

# Run similarity search in thread
def run_similarity():
    global model_type, features_file, isTextSearch
    
    # Get model, dataset folder and input image
    current_model_type = model_type
    input_image_path = input_image_entry.get().strip()
    dataset_folder_path = dataset_folder_entry.get().strip()
    isTextSearch = False
    
    # Get similarity threshold and number of similar images
    similarity_threshold = float(threshold_entry.get()) if threshold_entry.get().strip() else 0.2
    num_similar_images = int(top_k_entry.get()) if top_k_entry.get().isdigit() else 5
    
    # Get model and run similarity
    model, transform = setup_model_and_transform(current_model_type)
    similar_images = find_similar_images(input_image_path, dataset_folder_path, features_file, model, transform, num_similar_images, similarity_threshold, current_model_type)
    
    # Update GUI with results
    update_gui_with_similar_images(similar_images)

# Get and plot similar images
def get_and_plot_similar_images():
    threading.Thread(target=run_similarity, daemon=True).start()

# Run clustering in thread
def run_clustering():
    global model_type, features_file, clusters, current_cluster_index

    current_model_type = model_type
    # Get dataset folder
    dataset_folder_path = dataset_folder_entry.get().strip()
    
    # Get similarity threshold and Max number of clusters
    similarity_threshold = float(threshold_entry.get()) if threshold_entry.get().strip() else None
    max_clusters = int(max_num_clusters_entry.get()) if max_num_clusters_entry.get().isdigit() else None
    
    min_images_cluster = int(min_images_per_cluster_entry.get()) if min_images_per_cluster_entry.get().isdigit() else None
    max_images_cluster = int(max_images_per_cluster_entry.get()) if max_images_per_cluster_entry.get().isdigit() else None
    
    model, transform = setup_model_and_transform(current_model_type)
    current_cluster_index = 0
    clusters = cluster_similar_images(dataset_folder_path, features_file, model, transform, max_clusters, similarity_threshold, min_images_cluster, max_images_cluster, current_model_type)
    
    # Update GUI with results
    update_gui_with_clusters(clusters)

# Update GUI after clustering
def update_gui_with_clusters(clusters_result):
    global root, current_cluster_index
    if clusters_result:
        root.after(0, lambda: plot_image_clusters_scrollable(clusters_result, cluster_index=current_cluster_index))
    else:
        root.after(0, lambda: messagebox.showinfo("Info", "No image clusters found."))

# Get and plot image clusters
def get_and_plot_image_clusters():
    threading.Thread(target=run_clustering, daemon=True).start()

# Get next cluster when "Next Cluster" button is clicked
def next_cluster():
    global current_cluster_index
    if clusters and current_cluster_index < len(clusters) - 1:
        current_cluster_index += 1
        update_cluster_plot(clusters, cluster_index=current_cluster_index)

# Get previous cluster when "Previous Cluster" button is clicked
def previous_cluster():
    global current_cluster_index
    if clusters and current_cluster_index > 0:
        current_cluster_index -= 1
        update_cluster_plot(clusters, cluster_index=current_cluster_index)

#-------------------------- CustomTkinter GUI -----------------------------#

def create_gui(root_window):
    global root, dataset_folder_entry, text_search_entry, input_image_entry
    global threshold_entry, top_k_entry, max_num_clusters_entry
    global min_images_per_cluster_entry, max_images_per_cluster_entry
    
    root = root_window
    
    # Set theme and color
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    # Configure main window
    root.title("Image Similarity & Clustering")
    root.geometry("700x740")
    
    # Create main scrollable frame
    main_frame = ctk.CTkScrollableFrame(root, width=950, height=700)
    main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
    
    # Configure grid
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # -------------------------- GUI Layout ------------------------------#

    # Model selection
    ctk.CTkLabel(main_frame, text="Model Type").pack(pady=(0, 2))
    model_type_var = ctk.StringVar(value="swin")
    model_select = ctk.CTkOptionMenu(
        main_frame,
        values=["swin", "clip"],
        variable=model_type_var,
        command=get_model_type,
        width=400,
    )
    model_select.pack(pady=(0, 8))
    
    # Dataset folder selection
    ctk.CTkLabel(main_frame, text="Dataset Folder").pack(pady=(0, 2))
    dataset_frame = ctk.CTkFrame(main_frame)
    dataset_frame.pack(fill="x", padx=5, pady=(0, 8))
    dataset_folder_entry = ctk.CTkEntry(dataset_frame, width=500)
    dataset_folder_entry.pack(side="left", padx=5)
    ctk.CTkButton(
        dataset_frame,
        text="Browse",
        command=get_dataset_folder,
        width=150
    ).pack(side="right", padx=5)
    
    # Text search
    ctk.CTkLabel(main_frame, text="Text Search").pack(pady=(0, 2))
    text_search_frame = ctk.CTkFrame(main_frame)
    text_search_frame.pack(fill="x", padx=5, pady=(0, 8))
    text_search_entry = ctk.CTkEntry(text_search_frame, width=500)
    text_search_entry.pack(side="left", padx=5)
    ctk.CTkButton(
        text_search_frame,
        text="Search",
        command=search_by_text,
        width=150
    ).pack(side="right", padx=5)
    
    # Input image selection
    ctk.CTkLabel(main_frame, text="Input Image").pack(pady=(0, 2))
    input_image_frame = ctk.CTkFrame(main_frame)
    input_image_frame.pack(fill="x", padx=5, pady=(0, 8))
    input_image_entry = ctk.CTkEntry(input_image_frame, width=500)
    input_image_entry.pack(side="left", padx=5)
    ctk.CTkButton(
        input_image_frame,
        text="Browse",
        command=get_image_path,
        width=150
    ).pack(side="right", padx=5)
    
    # Similarity parameters
    ctk.CTkLabel(main_frame, text="Similarity Threshold").pack(pady=(0, 2))
    threshold_entry = ctk.CTkEntry(main_frame, width=400)
    threshold_entry.pack(pady=(0, 8))
    
    ctk.CTkLabel(main_frame, text="Number of Similar Images").pack(pady=(0, 2))
    top_k_entry = ctk.CTkEntry(main_frame, width=400)
    top_k_entry.pack(pady=(0, 8))
    
    # Clustering parameters
    ctk.CTkLabel(main_frame, text="Max Number of Clusters").pack(pady=(0, 2))
    max_num_clusters_entry = ctk.CTkEntry(main_frame, width=400)
    max_num_clusters_entry.pack(pady=(0, 8))
    
    ctk.CTkLabel(main_frame, text="Min Images per Cluster").pack(pady=(0, 2))
    min_images_per_cluster_entry = ctk.CTkEntry(main_frame, width=400)
    min_images_per_cluster_entry.pack(pady=(0, 8))
    
    ctk.CTkLabel(main_frame, text="Max Images per Cluster").pack(pady=(0, 2))
    max_images_per_cluster_entry = ctk.CTkEntry(main_frame, width=400)
    max_images_per_cluster_entry.pack(pady=(0, 8))
    
    # Action buttons
    action_frame = ctk.CTkFrame(main_frame)
    action_frame.pack(pady=8)
    
    ctk.CTkButton(
        action_frame,
        text="Find Similar Images",
        command=get_and_plot_similar_images,
        width=190
    ).pack(side="left", padx=5)
    
    ctk.CTkButton(
        action_frame,
        text="Cluster Images",
        command=get_and_plot_image_clusters,
        width=190
    ).pack(side="left", padx=5)
    
    # Navigation buttons
    nav_frame = ctk.CTkFrame(main_frame)
    nav_frame.pack(pady=8)
    
    ctk.CTkButton(
        nav_frame,
        text="Previous Cluster",
        command=previous_cluster,
        width=190
    ).pack(side="left", padx=5)
    
    ctk.CTkButton(
        nav_frame,
        text="Next Cluster",
        command=next_cluster,
        width=190
    ).pack(side="left", padx=5)

if __name__ == "__main__":
    root = ctk.CTk()
    create_gui(root)
    root.mainloop()