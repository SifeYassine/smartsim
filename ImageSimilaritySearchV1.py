import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from timm import create_model
from transformers import CLIPProcessor, CLIPModel
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

# Constants
BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)
SWIN_MODEL_NAME = 'swin_base_patch4_window7_224'
CLIP_VIT_MODEL_NAME = 'openai/clip-vit-base-patch32'
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# Global variables
features_file = None
current_cluster_index = 0
clusters = None
model = None
transform = None
model_type = None
clip_processor = None
isTextSearch = False

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

# Find similar images based for either a text prompt or an input image
def find_similar_images(query, dataset_folder, features_file, model, transform, top_k, similarity_threshold, model_type):
    # Get query features
    if isinstance(query, str):            
        # Process text query
        inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            query_features = model.get_text_features(**inputs)
            query_features = F.normalize(query_features, p=2, dim=1)
    else:
        # Process image query
        image_to_transform = Image.open(query).convert('RGB')
        input_tensor = transform(image_to_transform).unsqueeze(0)
        
        with torch.no_grad():
            if model_type == 'swin':
                query_features = model(input_tensor)
            else:  # clip
                query_features = model.get_image_features(input_tensor)
            query_features = F.normalize(query_features, p=2, dim=1)
    
    # Convert to numpy
    query_features = query_features.cpu().numpy()
    query_features_reshaped = query_features.reshape(1, -1)
    
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
def cluster_similar_images(dataset_folder, features_file, model, transform, max_num_clusters, similarity_threshold, model_type):
    
    features, image_paths = update_features(dataset_folder, features_file, model, transform, model_type)
    features_reshaped = np.array(features).reshape(len(features), -1)
    
    similarity_matrix = cosine_similarity(features_reshaped)
    distance_matrix = 1 - similarity_matrix

    # Apply max_num_clusters or similarity_threshold filtering if provided
    if max_num_clusters is not None and similarity_threshold is None:
        clustering = AgglomerativeClustering(n_clusters=max_num_clusters, metric='precomputed', linkage='average', distance_threshold=None)
    elif max_num_clusters is None and similarity_threshold is not None:
        clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=1 - similarity_threshold)

    labels = clustering.fit_predict(distance_matrix)

    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(image_paths[i])

    return clusters

#-------------------------- Plotting Logic -----------------------------#

# Get and plot similar images
def plot_similar_images(similar_images):
    if not similar_images:
        return
    
    # Close previous plot if it exists
    plt.close()
    
    def plot():
        num_similar_images = len(similar_images)
        num_cols = min(num_similar_images, 4)
        num_rows = (num_similar_images + num_cols - 1) // num_cols
        
        plt.figure(figsize=(8, 3 * num_rows))
        plt.suptitle('Similar Images to Input Image', fontsize=12)
        
        for i, (img_path, similarity) in enumerate(similar_images):
            image = Image.open(img_path)
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(image)
            plt.title(f'Similarity: {similarity*100:.2f}%' if i > 0 else 'Input Image')
            plt.axis('off')
        
        plt.tight_layout()
    
    return plot

# Get and Plot image clusters
def plot_image_clusters(clusters, cluster_index=0):
    if not clusters:
        return
    
    # Close previous plot if it exists
    plt.close()

    def plot():
        plt.figure(figsize=(8, 6))
        cluster_label = list(clusters.keys())[cluster_index]
        cluster_images = clusters[cluster_label]
        
        num_images = len(cluster_images)
        num_cols = min(num_images, 4)
        num_rows = (num_images + num_cols - 1) // num_cols

        plt.suptitle(f"Cluster {cluster_index + 1}/{len(clusters)}", fontsize=12)

        for j, image_path in enumerate(cluster_images):
            image = Image.open(image_path)
            plt.subplot(num_rows, num_cols, j + 1)
            plt.imshow(image)
            plt.axis('off')

        plt.tight_layout()

    return plot

#-------------------------- Tkinter GUI -----------------------------#

def create_gui(root):
    root.title("Image Similarity & Clustering")
    root.geometry("800x450")
    
    # Plot the figure
    def plot_the_figure(plot_function):
        if plot_function is None:
            return

        plot_function()
        plt.show()
    
    # Get model type        
    def get_model_type(model_type_var):
        global model_type
        model_type = model_type_var.get().strip()   
    
    # Get dataset folder and update features_file
    def get_dataset_folder(dataset_folder_entry):
        global features_file, model_type
        dataset_folder = filedialog.askdirectory()
        
        if dataset_folder:
            folder_name = os.path.basename(os.path.normpath(dataset_folder))
            # Set the features file name based on the dataset folder's name
            features_file = f'{folder_name}_dataset_features_{model_type}.pkl'
            dataset_folder_entry.delete(0, tk.END)
            dataset_folder_entry.insert(0, dataset_folder)
            
            print(f"Features file set to: {features_file}")
    
    # Get input image
    def get_image_path(input_image_entry):
        input_image = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.jfif")])
        if input_image:
            input_image_entry.delete(0, tk.END)
            input_image_entry.insert(0, input_image)

    # Text search function
    def search_by_text():
        def run_text_search():
            global model_type, features_file, isTextSearch
            
            # Get model, dataset folder and input image
            current_model_type = model_type
            print(f"Current model type: {current_model_type}")
            dataset_folder_path = dataset_folder.get().strip()
            prompt = text_search.get().strip()
            isTextSearch = True
            
            # Get similarity threshold and number of similar images
            similarity_threshold = float(threshold.get()) if threshold.get().strip() else 0.12
            num_similar_images = int(top_k.get()) if top_k.get().isdigit() else 5
            
            # Get model and run text search
            model, transform = setup_model_and_transform(current_model_type)
            
            similar_images = find_similar_images(prompt, dataset_folder_path, features_file, model, transform, num_similar_images, similarity_threshold, current_model_type)
            
          # Update GUI with results
            def update_gui():
                if similar_images:
                    plot_function = plot_similar_images(similar_images)
                    plot_the_figure(plot_function)
                else:
                    messagebox.showinfo("Info", "No similar images found.")
            
            root.after(0, update_gui)
        
        threading.Thread(target=run_text_search, daemon=True).start()
    
    # Get and plot similar images
    def get_and_plot_similar_images():
        def run_similarity():
            global model_type, features_file
            
            # Get model, dataset folder and input image
            current_model_type = model_type
            input_image_path = input_image.get().strip()
            dataset_folder_path = dataset_folder.get().strip()
            
            # Get similarity threshold and number of similar images
            similarity_threshold = float(threshold.get()) if threshold.get().strip() else 0.2
            num_similar_images = int(top_k.get()) if top_k.get().isdigit() else 5
            
            # Get model and run similarity
            model, transform = setup_model_and_transform(current_model_type)
            similar_images = find_similar_images(input_image_path, dataset_folder_path, features_file, model, transform, num_similar_images, similarity_threshold, current_model_type)
            
            # Update GUI with results
            def update_gui():
                if similar_images:
                    plot_function = plot_similar_images(similar_images)
                    plot_the_figure(plot_function)
                else:
                    messagebox.showinfo("Info", "No similar images found.")
            
            root.after(0, update_gui)
        
        threading.Thread(target=run_similarity, daemon=True).start()
    
    # Get and plot image clusters
    def get_and_plot_image_clusters():
        def run_clustering():
            # Get dataset folder
            dataset_folder_path = dataset_folder.get().strip()
            
            # Get Min number of images per cluster and Max number of clusters
            similarity_threshold = float(threshold.get()) if threshold.get().strip() else None
            max_clusters = int(max_num_clusters.get()) if max_num_clusters.get().isdigit() else None
            
            # Get model and run clustering
            current_model_type = model_type_entry.get()
            model, transform = setup_model_and_transform(current_model_type)
            global clusters, current_cluster_index
            current_cluster_index = 0
            clusters = cluster_similar_images(dataset_folder_path, features_file, model, transform, max_clusters, similarity_threshold, current_model_type)
            
            # Update GUI with results
            def update_gui():
                if clusters:
                    plot_function = plot_image_clusters(clusters, cluster_index=0)
                    plot_the_figure(plot_function)
                else:
                    messagebox.showinfo("Info", "No image clusters found.")
            
            root.after(0, update_gui)
        
        threading.Thread(target=run_clustering, daemon=True).start()
    
    # Get next cluster when "Next Cluster" button is clicked
    def next_cluster():
        global current_cluster_index
        if clusters and current_cluster_index < len(clusters) - 1:
            current_cluster_index += 1
            plot_function = plot_image_clusters(clusters, cluster_index=current_cluster_index)
            plot_the_figure(plot_function)
    
    # Get previous cluster when "Previous Cluster" button is clicked
    def previous_cluster():
        global current_cluster_index
        if clusters and current_cluster_index > 0:
            current_cluster_index -= 1
            plot_function = plot_image_clusters(clusters, cluster_index=current_cluster_index)
            plot_the_figure(plot_function)
    
    #-------------------------- GUI Layout ------------------------------#
    
    # Model selection
    tk.Label(root, text="Model Type:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
    model_type_entry = tk.StringVar(root)
    model_select = ttk.Combobox(root, textvariable=model_type_entry, values=["swin", "clip"], state="readonly")
    model_type_entry.trace_add("write", lambda *args: get_model_type(model_type_entry))
    model_select.grid(row=0, column=1, padx=10, pady=10)
    
    # Dataset folder selection
    tk.Label(root, text="Dataset Folder:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
    dataset_folder = tk.Entry(root, width=40)
    dataset_folder.grid(row=1, column=1, padx=10, pady=10)
    tk.Button(root, text="Browse", command=lambda: get_dataset_folder(dataset_folder)).grid(row=1, column=2, padx=10, pady=10)

    # Text search input
    tk.Label(root, text="Text Search:").grid(row=2, column=0, padx=10, pady=10, sticky="e")
    text_search = tk.Entry(root, width=40)
    text_search.grid(row=2, column=1, padx=10, pady=10)
    tk.Button(root, text="Search by Text", command=search_by_text).grid(row=2, column=2, padx=10, pady=10)
    
    # Input image selection
    tk.Label(root, text="Input Image:").grid(row=3, column=0, padx=10, pady=10, sticky="e")
    input_image = tk.Entry(root, width=40)
    input_image.grid(row=3, column=1, padx=10, pady=10)
    tk.Button(root, text="Browse", command=lambda: get_image_path(input_image)).grid(row=3, column=2, padx=10, pady=10)
    
    # Similarity threshold
    tk.Label(root, text="Similarity Threshold (None if Max Num Cltrs):").grid(row=4, column=0, padx=10, pady=10, sticky="e")
    threshold = tk.Entry(root)
    threshold.grid(row=4, column=1, padx=10, pady=10)

    # Number of similar images (top_k)
    tk.Label(root, text="Number of Similar Images:").grid(row=5, column=0, padx=10, pady=10, sticky="e")
    top_k = tk.Entry(root)
    top_k.grid(row=5, column=1, padx=10, pady=10)

    # Max Number of Clusters
    tk.Label(root, text="Max Num of Clusters (None if Sim Thresh):").grid(row=8, column=0, padx=10, pady=10, sticky="e")
    max_num_clusters = tk.Entry(root)
    max_num_clusters.grid(row=8, column=1, padx=10, pady=10)

    # Action buttons
    action_frame = tk.Frame(root)
    action_frame.grid(row=9, column=1, padx=20, pady=10)

    tk.Button(action_frame, text="Find Similar Images", command=get_and_plot_similar_images).grid(row=0, column=0, padx=10)
    tk.Button(action_frame, text="Cluster Images", command=get_and_plot_image_clusters).grid(row=0, column=1, padx=10)

    # Navigation buttons
    nav_frame = tk.Frame(root)
    nav_frame.grid(row=10, column=1, padx=20, pady=10)

    tk.Button(nav_frame, text="Previous Cluster", command=previous_cluster).grid(row=0, column=0, padx=10)
    tk.Button(nav_frame, text="Next Cluster", command=next_cluster).grid(row=0, column=1, padx=10)

if __name__ == "__main__":
    root = tk.Tk()
    create_gui(root)
    root.mainloop()