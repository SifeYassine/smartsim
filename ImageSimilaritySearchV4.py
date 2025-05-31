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

# Image display constants
IMAGES_PER_ROW = 4
IMAGE_DISPLAY_SIZE = (2.5, 2.5)

# Global variables
features_file = None
current_cluster_index = 0
current_plot_window = None
current_canvas = None
current_scrollable_frame = None
current_figure = None
current_canvas_widget = None
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

#-------------------------- Scrollable Plotting Logic -----------------------------#

# Event handler for frame configuration
def on_frame_configure(event, canvas):
    canvas.configure(scrollregion=canvas.bbox('all'))

# Event handler for canvas configuration  
def on_canvas_configure(event, canvas, canvas_window):
    canvas_width = event.width
    canvas.itemconfig(canvas_window, width=canvas_width)

# Event handler for mouse wheel scrolling
def on_mousewheel(event, canvas):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

# Create scrollable plot window and return necessary components
def create_scrollable_plot_window(title="Image Results"):
    window = ctk.CTkToplevel()
    window.title(title)
    window.geometry("1000x800")
    
    # Create main frame
    main_frame = ctk.CTkFrame(window)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Create tkinter canvas and scrollbar (for matplotlib integration)
    canvas = tk.Canvas(main_frame)
    scrollbar = ctk.CTkScrollbar(main_frame, orientation="vertical", command=canvas.yview)
    
    # Configure canvas scrolling
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Pack scrollbars and canvas
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    
    # Create scrollable frame
    scrollable_frame = ctk.CTkFrame(canvas)
    canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
    
    # Bind events using lambda to pass additional parameters
    scrollable_frame.bind('<Configure>', lambda event: on_frame_configure(event, canvas))
    canvas.bind('<Configure>', lambda event: on_canvas_configure(event, canvas, canvas_window))
    canvas.bind_all('<MouseWheel>', lambda event: on_mousewheel(event, canvas))
    
    return window, canvas, scrollable_frame

# Create matplotlib plot within scrollable frame
def create_scrollable_plot(canvas, scrollable_frame, images_data, title, is_similarity=False):
    global current_figure, current_canvas_widget
    
    if not images_data:
        return
        
    num_images = len(images_data)
    num_rows = (num_images + IMAGES_PER_ROW - 1) // IMAGES_PER_ROW
    
    # Calculate figure size based on number of images and fixed image size
    fig_width = IMAGES_PER_ROW * IMAGE_DISPLAY_SIZE[0]
    fig_height = num_rows * IMAGE_DISPLAY_SIZE[1]
    
    # Create matplotlib figure
    current_figure = Figure(figsize=(fig_width, fig_height), dpi=100)
    current_figure.suptitle(title, fontsize=14, y=0.98)
    
    # Plot images
    for i, data in enumerate(images_data):
        if is_similarity:
            img_path, similarity = data
            title_text = f'Similarity: {similarity*100:.2f}%' 
        else:
            img_path = data
            title_text = os.path.basename(img_path)
        
        try:
            image = Image.open(img_path)
            ax = current_figure.add_subplot(num_rows, IMAGES_PER_ROW, i + 1)
            ax.imshow(image)
            ax.set_title(title_text, fontsize=10, pad=5)
            ax.axis('off')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    current_figure.tight_layout()
    
    # Embed in tkinter
    if current_canvas_widget:
        current_canvas_widget.get_tk_widget().destroy()
        
    current_canvas_widget = FigureCanvasTkAgg(current_figure, scrollable_frame)
    current_canvas_widget.draw()
    current_canvas_widget.get_tk_widget().pack(fill="both", expand=True)
    
    # Update scroll region
    scrollable_frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox('all'))

# Update plot within existing scrollable window
def update_scrollable_plot(images_data, title, is_similarity=False):
    global current_canvas, current_scrollable_frame
    
    if current_canvas and current_scrollable_frame:
        create_scrollable_plot(current_canvas, current_scrollable_frame, images_data, title, is_similarity)

# Plot similar images in a scrollable window
def plot_similar_images_scrollable(similar_images):
    if not similar_images:
        return
    
    global current_plot_window, current_canvas, current_scrollable_frame
    
    # Close previous window if it exists
    if current_plot_window and current_plot_window.winfo_exists():
        current_plot_window.destroy()
    
    # Create new scrollable plot window
    current_plot_window, current_canvas, current_scrollable_frame = create_scrollable_plot_window("Similar Images")
    create_scrollable_plot(current_canvas, current_scrollable_frame, similar_images, "Similar Images to Query", is_similarity=True)

# Plot image clusters in a scrollable window
def plot_image_clusters_scrollable(clusters, cluster_index=0):
    if not clusters:
        return
    
    global current_plot_window, current_canvas, current_scrollable_frame
    
    # Close previous window if it exists
    if current_plot_window and current_plot_window.winfo_exists():
        current_plot_window.destroy()
    
    cluster_label = list(clusters.keys())[cluster_index]
    cluster_images = clusters[cluster_label]
    
    title = f"Cluster {cluster_index + 1}/{len(clusters)} ({len(cluster_images)} images)"
    
    # Create new scrollable plot window
    current_plot_window, current_canvas, current_scrollable_frame = create_scrollable_plot_window("Image Clusters")
    create_scrollable_plot(current_canvas, current_scrollable_frame, cluster_images, title, is_similarity=False)

# Update cluster plot within existing window (for navigation)
def update_cluster_plot(clusters, cluster_index=0):
    if not clusters:
        return
    
    cluster_label = list(clusters.keys())[cluster_index]
    cluster_images = clusters[cluster_label]
    
    title = f"Cluster {cluster_index + 1}/{len(clusters)} ({len(cluster_images)} images)"
    
    # Update plot within existing window
    update_scrollable_plot(cluster_images, title, is_similarity=False)

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
            
          # Update GUI with results using efficient scrollable plot
            def update_gui():
                if similar_images:
                    plot_similar_images_scrollable(similar_images)
                else:
                    messagebox.showinfo("Info", "No similar images found.")
            
            root.after(0, update_gui)
        
        threading.Thread(target=run_text_search, daemon=True).start()
    
    # Get and plot similar images
    def get_and_plot_similar_images():
        def run_similarity():
            global model_type, features_file, isTextSearch
            
            # Get model, dataset folder and input image
            current_model_type = model_type
            input_image_path = input_image.get().strip()
            dataset_folder_path = dataset_folder.get().strip()
            isTextSearch = False
            
            # Get similarity threshold and number of similar images
            similarity_threshold = float(threshold.get()) if threshold.get().strip() else 0.2
            num_similar_images = int(top_k.get()) if top_k.get().isdigit() else 5
            
            # Get model and run similarity
            model, transform = setup_model_and_transform(current_model_type)
            similar_images = find_similar_images(input_image_path, dataset_folder_path, features_file, model, transform, num_similar_images, similarity_threshold, current_model_type)
            
            # Update GUI with results using efficient scrollable plot
            def update_gui():
                if similar_images:
                    plot_similar_images_scrollable(similar_images)
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
            current_cluster_index = 0
            clusters = cluster_similar_images(dataset_folder_path, features_file, model, transform, max_clusters, similarity_threshold, min_images_cluster, max_images_cluster, current_model_type)
            
            def update_gui():
                if clusters:
                    plot_image_clusters_scrollable(clusters, cluster_index=current_cluster_index)
                else:
                    messagebox.showinfo("Info", "No image clusters found.")
            
            root.after(0, update_gui)
        
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