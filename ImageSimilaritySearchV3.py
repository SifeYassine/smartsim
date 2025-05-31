import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from timm import create_model
from transformers import CLIPProcessor, CLIPModel
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import customtkinter as ctk
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
current_plot_window = None
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

# Get and plot similar images
def plot_similar_images(similar_images):
    if not similar_images:
        return None

    num_images = len(similar_images)
    num_cols = min(num_images, 4)
    num_rows = (num_images + num_cols - 1) // num_cols

    # Adjust figure size for fixed image dimensions
    fig = Figure(figsize=(8, 2 * num_rows))
    fig.suptitle('Similar Images to Input Image', fontsize=12)

    for i, (img_path, similarity) in enumerate(similar_images):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        image = Image.open(img_path)
        ax.imshow(image)
        ax.set_title(f'Similarity: {similarity*100:.2f}%' if i > 0 else 'Input Image')
        ax.axis('off')

    fig.tight_layout()
    return fig

# Get and Plot image clusters
def plot_image_clusters(clusters, cluster_index):
    if not clusters:
        return None

    cluster_label = list(clusters.keys())[cluster_index]
    cluster_images = clusters[cluster_label]
    num_images = len(cluster_images)
    num_cols = min(num_images, 4)
    num_rows = (num_images + num_cols - 1) // num_cols

    fig = Figure(figsize=(8, 2 * num_rows))
    fig.suptitle(f"Cluster {cluster_index + 1}/{len(clusters)}", fontsize=12)

    for j, image_path in enumerate(cluster_images):
        ax = fig.add_subplot(num_rows, num_cols, j + 1)
        image = Image.open(image_path)
        ax.imshow(image)
        ax.axis('off')

    fig.tight_layout()
    return fig

#-------------------------- Tkinter GUI -----------------------------#

def create_gui(root):
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
    
    # Plot the figure
    def display_scrollable_figure(root, fig, title):
        global current_plot_window

        # Close previous plot window if it exists
        if current_plot_window is not None:
            current_plot_window.destroy()

        # Create new window
        current_plot_window = ctk.CTkToplevel(root)
        current_plot_window.title(title)
        current_plot_window.geometry("800x600")

        # Main frame to hold canvas and scrollbar
        main_frame = ctk.CTkFrame(current_plot_window)
        main_frame.pack(fill="both", expand=True)

        # Tkinter canvas and scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ctk.CTkScrollbar(main_frame, orientation="vertical", command=canvas.yview)
        scrollable_frame = ctk.CTkFrame(canvas)

        # Configure canvas scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack elements
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Embed the figure
        canvas_widget = FigureCanvasTkAgg(fig, master=scrollable_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill="both", expand=True)
    
    # Get model type
    def get_model_type(choice):
        global model_type
        model_type = choice
    
    # Get dataset folder and update features_file
    def get_dataset_folder(dataset_folder_entry):
        global features_file, model_type
        dataset_folder = filedialog.askdirectory()
        
        if dataset_folder:
            folder_name = os.path.basename(os.path.normpath(dataset_folder))
            # Set the features file name based on the dataset folder's name
            features_file = f'{folder_name}_dataset_features_{model_type}.pkl'
            dataset_folder_entry.delete(0, ctk.END)
            dataset_folder_entry.insert(0, dataset_folder)
            
            print(f"Features file set to: {features_file}")
    
    # Get input image
    def get_image_path(input_image_entry):
        input_image = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.jfif")])
        if input_image:
            input_image_entry.delete(0, ctk.END)
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
                    fig = plot_similar_images(similar_images)
                    if fig:
                        display_scrollable_figure(root, fig, "Similar Images")
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
                    fig = plot_similar_images(similar_images)
                    if fig:
                        display_scrollable_figure(root, fig, "Similar Images")
                    else:
                        messagebox.showinfo("Info", "No similar images found.")
            
            root.after(0, update_gui)
        
        threading.Thread(target=run_similarity, daemon=True).start()
    
    # Get and plot image clusters
    def get_and_plot_image_clusters():
        def run_clustering():
            global model_type, features_file, clusters, current_cluster_index

            current_model_type = model_type
            # Get dataset folder
            dataset_folder_path = dataset_folder.get().strip()
            
            # Get similarity threshold and Max number of clusters
            similarity_threshold = float(threshold.get()) if threshold.get().strip() else None
            max_clusters = int(max_num_clusters.get()) if max_num_clusters.get().isdigit() else None
            

            min_images_cluster = int(min_images_per_cluster.get()) if min_images_per_cluster.get().isdigit() else None
            max_images_cluster = int(max_images_per_cluster.get()) if max_images_per_cluster.get().isdigit() else None
            
            model, transform = setup_model_and_transform(current_model_type)
            clusters = cluster_similar_images(dataset_folder_path, features_file, model, transform, max_clusters, similarity_threshold, min_images_cluster, max_images_cluster, current_model_type)
            
            def update_gui():
                if clusters:
                    fig = plot_image_clusters(clusters, cluster_index=current_cluster_index)
                    if fig:
                        display_scrollable_figure(root, fig, f"Cluster {current_cluster_index + 1}")
                    else:
                        messagebox.showinfo("Info", "No image clusters found.")
            
            root.after(0, update_gui)
        
        threading.Thread(target=run_clustering, daemon=True).start()
    
    # Get next cluster when "Next Cluster" button is clicked
    def next_cluster():
        global current_cluster_index, current_plot_window
        if clusters and current_cluster_index < len(clusters) - 1:
            current_cluster_index += 1
            fig = plot_image_clusters(clusters, cluster_index=current_cluster_index)
            display_scrollable_figure(root, fig, f"Cluster {current_cluster_index + 1}")
    
    # Get previous cluster when "Previous Cluster" button is clicked
    def previous_cluster():
        global current_cluster_index, current_plot_window
        if clusters and current_cluster_index > 0:
            current_cluster_index -= 1
            fig = plot_image_clusters(clusters, cluster_index=current_cluster_index)
            display_scrollable_figure(root, fig, f"Cluster {current_cluster_index + 1}")


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
    dataset_folder = ctk.CTkEntry(dataset_frame, width=500)
    dataset_folder.pack(side="left", padx=5)
    ctk.CTkButton(
        dataset_frame,
        text="Browse",
        command=lambda: get_dataset_folder(dataset_folder),
        width=150
    ).pack(side="right", padx=5)
    
    # Text search
    ctk.CTkLabel(main_frame, text="Text Search").pack(pady=(0, 2))
    text_search_frame = ctk.CTkFrame(main_frame)
    text_search_frame.pack(fill="x", padx=5, pady=(0, 8))
    text_search = ctk.CTkEntry(text_search_frame, width=500)
    text_search.pack(side="left", padx=5)
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
    input_image = ctk.CTkEntry(input_image_frame, width=500)
    input_image.pack(side="left", padx=5)
    ctk.CTkButton(
        input_image_frame,
        text="Browse",
        command=lambda: get_image_path(input_image),
        width=150
    ).pack(side="right", padx=5)
    
    # Similarity parameters
    ctk.CTkLabel(main_frame, text="Similarity Threshold").pack(pady=(0, 2))
    threshold = ctk.CTkEntry(main_frame, width=400)
    threshold.pack(pady=(0, 8))
    
    ctk.CTkLabel(main_frame, text="Number of Similar Images").pack(pady=(0, 2))
    top_k = ctk.CTkEntry(main_frame, width=400)
    top_k.pack(pady=(0, 8))
    
    # Clustering parameters
    ctk.CTkLabel(main_frame, text="Max Number of Clusters").pack(pady=(0, 2))
    max_num_clusters = ctk.CTkEntry(main_frame, width=400)
    max_num_clusters.pack(pady=(0, 8))
    
    ctk.CTkLabel(main_frame, text="Min Images per Cluster").pack(pady=(0, 2))
    min_images_per_cluster = ctk.CTkEntry(main_frame, width=400)
    min_images_per_cluster.pack(pady=(0, 8))
    
    ctk.CTkLabel(main_frame, text="Max Images per Cluster").pack(pady=(0, 2))
    max_images_per_cluster = ctk.CTkEntry(main_frame, width=400)
    max_images_per_cluster.pack(pady=(0, 8))
    
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