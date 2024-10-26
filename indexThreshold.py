import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import matplotlib.pyplot as plt
from timm import create_model
import pickle
from tqdm import tqdm

# Parameters
dataset_folder = './Test'
features_file = './Test_dataset_features.pkl'
SIMILARITY_THRESHOLD = 0.2  # Control the similarity threshold

# Load a pre-trained Swin Transformer model
model = create_model('swin_base_patch4_window7_224', pretrained=True)
model.head = torch.nn.Identity()
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features from an image
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    features = features.squeeze()
    return features.numpy()

# Function to save features to a file
def save_features(features, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(features, f)

# Function to load features from a file
def load_features(file_path):
    with open(file_path, 'rb') as f:
        features = pickle.load(f)
    return features

# Function to calculate cosine similarity matrix
def calculate_cosine_similarity(features):
    features = np.array(features)
    features_flattened = features.reshape(features.shape[0], -1)
    similarity_matrix = cosine_similarity(features_flattened)
    return similarity_matrix

# Function to cluster images using hierarchical clustering    
def hierarchical_clustering(dataset_features, image_paths, similarity_threshold):
    similarity_matrix = calculate_cosine_similarity(dataset_features)
    distance_matrix = 1 - similarity_matrix

    clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=1 - similarity_threshold)
    labels = clustering.fit_predict(distance_matrix)

    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(image_paths[i])

    return clusters

# Load or extract features from the dataset images
if os.path.exists(features_file):
    dataset_features, image_paths = load_features(features_file)
else:
    dataset_features = []
    image_paths = []
    image_files = os.listdir(dataset_folder)
    for img_name in tqdm(image_files, desc="Extracting features from dataset images"):
        img_path = os.path.join(dataset_folder, img_name)
        features = extract_features(img_path)
        dataset_features.append(features)
        image_paths.append(img_path)
    save_features((dataset_features, image_paths), features_file)

# Get the clusters
clusters = hierarchical_clustering(dataset_features, image_paths, similarity_threshold=SIMILARITY_THRESHOLD)

# Display the clusters
for cluster_id, image_paths in clusters.items():
    print(f"Cluster {cluster_id}:")
    for img_path in image_paths:
        print(img_path)
    print()

    num_images = len(image_paths)
    num_rows = (num_images // 5) + 1
    
    fig, axes = plt.subplots(num_rows, 5, figsize=(10, 4*num_rows))
    axes = axes.flatten()

    # Display cluster name at the top-center of the subplot
    fig.suptitle(f"Cluster {cluster_id}")

    for i, img_path in enumerate(image_paths):
        image = Image.open(img_path)
        axes[i].imshow(image)
        axes[i].axis('off')

    for j in range(i + 1, num_rows * 5):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()