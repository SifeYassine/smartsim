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
model_type = 'swin'

#-------------------------- Setup Logic -----------------------------#

# Load a pre-trained Swin Transformer model and set to evaluation mode
def setup_model_and_transform(model_type='swin'):
    global model, transform
    
    # Swin Model Setup
    if model_type == 'swin':
        print("Setting up Swin Transformer...")
        model = create_model(SWIN_MODEL_NAME, pretrained=True)
        model.head = torch.nn.Identity()
        
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])
    
    # CLIP Model Setup    
    elif model_type == 'clip':
        print("Setting up CLIP with safetensors...")
        model = CLIPModel.from_pretrained(CLIP_VIT_MODEL_NAME, use_safetensors=True)
        processor = CLIPProcessor.from_pretrained(CLIP_VIT_MODEL_NAME)
        
        def hf_preprocess(img):
            return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        transform = hf_preprocess
    
    model.eval()
    model = model.float()
    
    return model, transform

#-------------------------- Feature Extraction Logic -----------------------------#

# Extract features from a batch of images
def extract_features_batch(image_paths, model, transform, model_type='swin'):
    batch = [transform(Image.open(img_path).convert('RGB')) for img_path in image_paths]
    batch = torch.stack(batch)
    
    with torch.no_grad():
        if model_type == 'swin':
            features = model(batch)
        else:  # clip
            features = model.get_image_features(batch)
        
        features = F.normalize(features, p=2, dim=1)
        features = features.cpu().numpy()
    return features


# Extract features in parallel
def extract_features_parallel(image_paths, model, transform, model_type='swin', batch_size=BATCH_SIZE):
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
        return pickle.load(f)

# Update features and filenames in the pickle file if new images are added
def update_features(dataset_folder, features_file, model, transform, model_type='swin'):
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

# Modified to handle both models
def find_similar_images(input_image_path, dataset_folder, features_file, model, transform, top_k, similarity_threshold, model_type='swin'):
    image_to_transform = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(image_to_transform).unsqueeze(0)
    
    with torch.no_grad():
        if model_type == 'swin':
            input_features = model(input_tensor)
        else:  # clip
            input_features = model.get_image_features(input_tensor)
        
        input_features = F.normalize(input_features, p=2, dim=1)
        input_features = input_features.cpu().numpy()

    input_features_reshaped = input_features.reshape(1, -1)
    dataset_features, filenames = update_features(dataset_folder, features_file, model, transform, model_type)
    dataset_features_reshaped = np.array(dataset_features).reshape(len(dataset_features), -1)
    
    similarities = cosine_similarity(input_features_reshaped, dataset_features_reshaped)[0]
    filtered_indices = [i for i, sim in enumerate(similarities) if sim >= similarity_threshold]
    filtered_similarities = [similarities[i] for i in filtered_indices]
    
    if not filtered_similarities:
        return []
    
    sorted_indices = np.argsort(filtered_similarities)[-top_k-1:][::-1]
    return [(filenames[filtered_indices[idx]], filtered_similarities[idx]) for idx in sorted_indices]

# Modified to include model_type
def cluster_similar_images(dataset_folder, features_file, model, transform, min_images_per_cluster, max_num_clusters, model_type='swin'):
    features, image_paths = update_features(dataset_folder, features_file, model, transform, model_type)
    features_reshaped = np.array(features).reshape(len(features), -1)
    
    similarity_matrix = cosine_similarity(features_reshaped)
    distance_matrix = 1 - similarity_matrix
    
    if max_num_clusters is None:
        max_num_clusters = max(2, len(image_paths) // 5)
    
    thresholds = np.linspace(0.1, 0.9, 17)
    best_clusters = None
    best_score = float('-inf')
    
    for threshold in thresholds:
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,linkage='average')
        labels = clustering.fit_predict(distance_matrix)
        
        init_clusters = defaultdict(list)
        for i, label in enumerate(labels):
            init_clusters[label].append(image_paths[i])
        
        valid_clusters = {k: v for k, v in init_clusters.items() if len(v) >= min_images_per_cluster}
        num_valid_clusters = len(valid_clusters)
        
        if num_valid_clusters > 0 and num_valid_clusters <= max_num_clusters:
            average_cluster_size = np.mean([len(v) for v in valid_clusters.values()])
            score = num_valid_clusters * average_cluster_size
            
            if score > best_score:
                best_score = score
                best_clusters = valid_clusters

    if best_clusters is None or len(best_clusters) == 0:
        print("No suitable clusters found. Try adjusting the parameters.")
        return defaultdict(list)
    
    return best_clusters

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
    root.geometry("630x450")  # Slightly taller to accommodate model selection
    
    def plot_the_figure(plot_function):
        if plot_function is None:
            return

        plot_function()
        plt.show()

    def process_with_progress(process_function, handle_result, *args):            
            def run_process():
                result = process_function(*args)
                root.after(0, lambda: handle_result(result))
            
            threading.Thread(target=run_process, daemon=True).start()
            
    def get_model_type(model_type_var):
        global model_type
        model_type = model_type_var.get().strip()   
    
    # Modified to update model type
    def get_dataset_folder(dataset_folder_entry):
        global features_file, model_type
        dataset_folder = filedialog.askdirectory()
        
        if dataset_folder:
            folder_name = os.path.basename(os.path.normpath(dataset_folder))
            features_file = f'{folder_name}_dataset_features_{model_type}.pkl'
            dataset_folder_entry.delete(0, tk.END)
            dataset_folder_entry.insert(0, dataset_folder)
            
            print(f"Features file set to: {features_file}")
    
    def get_image_path(input_image_entry):
        input_image = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.jfif")])
        if input_image:
            input_image_entry.delete(0, tk.END)
            input_image_entry.insert(0, input_image)
    
    # Modified to use current model type
    def get_and_plot_similar_images():
        def run_similarity():
            input_image_path = input_image.get().strip()
            dataset_folder_path = dataset_folder.get().strip()
            
            similarity_threshold = float(threshold.get()) if threshold.get().strip() else 0.12
            num_similar_images = int(top_k.get()) if top_k.get().isdigit() else 5
            current_model_type = model_type_entry.get()
            
            model, transform = setup_model_and_transform(current_model_type)
            similar_images = find_similar_images(input_image_path, dataset_folder_path, features_file,
                                              model, transform, num_similar_images, similarity_threshold,
                                              current_model_type)
            
            def update_gui():
                if similar_images:
                    plot_function = plot_similar_images(similar_images)
                    plot_the_figure(plot_function)
                else:
                    messagebox.showinfo("Info", "No similar images found.")
            
            root.after(0, update_gui)
        
        threading.Thread(target=run_similarity, daemon=True).start()
    
    # Modified to use current model type
    def get_and_plot_image_clusters():
        def run_clustering():
            dataset_folder_path = dataset_folder.get().strip()
            
            min_cluster_size = int(min_num_images_per_cluster.get()) if min_num_images_per_cluster.get().isdigit() else 5
            max_clusters = int(max_num_clusters.get()) if max_num_clusters.get().isdigit() else 3
            current_model_type = model_type_entry.get()
            
            model, transform = setup_model_and_transform(current_model_type)
            global clusters, current_cluster_index
            current_cluster_index = 0
            clusters = cluster_similar_images(dataset_folder_path, features_file, model, transform,
                                           min_cluster_size, max_clusters, current_model_type)
            
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
    model_type_entry = tk.StringVar(value="swin")
    model_select = ttk.Combobox(root, textvariable=model_type_entry, values=["swin", "clip"], state="readonly")
    model_type_entry.trace_add("write", lambda *args: get_model_type(model_type_entry))  # Using trace_add instead of trace
    model_select.grid(row=0, column=1, padx=10, pady=10)
    
    # Dataset folder selection
    tk.Label(root, text="Dataset Folder:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
    dataset_folder = tk.Entry(root, width=40)
    dataset_folder.grid(row=1, column=1, padx=10, pady=10)
    tk.Button(root, text="Browse", command=lambda: get_dataset_folder(dataset_folder)).grid(row=1, column=2, padx=10, pady=10)

    # Input image selection (move to the next row)
    tk.Label(root, text="Input Image:").grid(row=2, column=0, padx=10, pady=10, sticky="e")
    input_image = tk.Entry(root, width=40)
    input_image.grid(row=2, column=1, padx=10, pady=10)
    tk.Button(root, text="Browse", command=lambda: get_image_path(input_image)).grid(row=2, column=2, padx=10, pady=10)

    # Update the rest of the rows accordingly
    # Similarity threshold input
    tk.Label(root, text="Similarity Threshold:").grid(row=3, column=0, padx=10, pady=10, sticky="e")
    threshold = tk.Entry(root)
    threshold.insert(0, "0.12")
    threshold.grid(row=3, column=1, padx=10, pady=10)

    # Number of similar images input (top_k)
    tk.Label(root, text="Number of Similar Images:").grid(row=4, column=0, padx=10, pady=10, sticky="e")
    top_k = tk.Entry(root)
    top_k.insert(0, "5")
    top_k.grid(row=4, column=1, padx=10, pady=10)

    # Min number of images per cluster
    tk.Label(root, text="Min Images per Cluster:").grid(row=5, column=0, padx=10, pady=10, sticky="e")
    min_num_images_per_cluster = tk.Entry(root)
    min_num_images_per_cluster.insert(0, "5")
    min_num_images_per_cluster.grid(row=5, column=1, padx=10, pady=10)

    # Max number of clusters
    tk.Label(root, text="Max Number of Clusters:").grid(row=6, column=0, padx=10, pady=10, sticky="e")
    max_num_clusters = tk.Entry(root)
    max_num_clusters.insert(0, "3")
    max_num_clusters.grid(row=6, column=1, padx=10, pady=10)

    # Action frame for buttons
    action_frame = tk.Frame(root)
    action_frame.grid(row=7, column=0, columnspan=3, padx=20, pady=10)

    tk.Button(action_frame, text="Find Similar Images", command=get_and_plot_similar_images).pack(side=tk.LEFT, padx=5)
    tk.Button(action_frame, text="Cluster Images", command=get_and_plot_image_clusters).pack(side=tk.LEFT, padx=5)

    # Navigation frame for navigation buttons
    nav_frame = tk.Frame(root)
    nav_frame.grid(row=8, column=0, columnspan=3, padx=20, pady=10)

    tk.Button(nav_frame, text="Previous Cluster", command=previous_cluster).pack(side=tk.LEFT, padx=5)
    tk.Button(nav_frame, text="Next Cluster", command=next_cluster).pack(side=tk.LEFT, padx=5)

if __name__ == "__main__":
    root = tk.Tk()
    create_gui(root)
    root.mainloop()