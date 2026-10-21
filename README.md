# SmartSim

A PyTorch desktop tool for **image similarity search** and **hierarchical clustering** of image collections, powered by pretrained vision models: **Swin Transformer** and **CLIP ViT**.

Point SmartSim at a folder of images and it can find the pictures most similar to a query image, find images matching a **natural-language description** (with CLIP), or automatically group the whole folder into clusters of visually similar images. Everything is driven from a simple CustomTkinter GUI.

## Features

- **Two backbone models**
  - `swin_base_patch4_window7_224` (via `timm`): strong visual features
  - `openai/clip-vit-base-patch32` (via Hugging Face `transformers`): enables text-to-image search
- **Image → image search**: return the top-K most similar images above a cosine-similarity threshold
- **Text → image search** (CLIP only): describe what you're looking for, e.g. *"a red car at night"*
- **Hierarchical clustering**: agglomerative clustering (average linkage, cosine distance) with either a fixed number of clusters or a similarity threshold, plus min/max cluster-size filters
- **Feature caching**: embeddings are extracted once in parallel batches and cached to a `.pkl` file per dataset and model; images added later are detected and embedded incrementally
- **Scrollable result viewer** with similarity scores and **Previous / Next** cluster navigation

## Tech Stack

- Python 3, PyTorch, torchvision
- `timm` (Swin Transformer), `transformers` (CLIP)
- scikit-learn (cosine similarity, AgglomerativeClustering)
- NumPy, Pillow, OpenCV
- CustomTkinter (GUI)

## Project Structure

```
smartsim/
├── ImageSimilaritySearchV7.py   # Latest version (refactored state management)
├── ImageSimilaritySearchV6.py   # Previous iteration
├── ImageSimilaritySearchV5.py   # Previous iteration
└── requirements.txt
```

> **V7** is the most recent and recommended version. V5 and V6 are kept for reference.

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
git clone https://github.com/SifeYassine/smartsim.git
cd smartsim

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
pip install opencv-python   # cv2 is imported but not listed in requirements.txt
```

### Configuration

The CLIP model is downloaded to a hard-coded cache directory:

```python
cache_dir=r'D:\AI Assets\CLIP'
```

Change this path in `setup_model_and_transform()` to a location that exists on your machine, or remove the argument to use the default Hugging Face cache.

### Run

```bash
python ImageSimilaritySearchV7.py
```

## Usage

1. **Model Type**: choose `swin` or `clip`.
2. **Dataset Folder**: browse to a folder of images (`.png`, `.jpg`, `.jpeg`, `.jfif`).
   The first run extracts and caches features to `<folder>_features_<model>.pkl`.
3. Then do one of the following:
   - **Find Similar Images**: pick an *Input Image*, then set the *Similarity Threshold* and *Number of Similar Images*.
   - **Text Search** *(CLIP)*: type a description and click **Search**.
   - **Cluster Images**: set *either* *Max Number of Clusters* **or** *Similarity Threshold*, optionally with *Min* **or** *Max Images per Cluster*, then browse results with **Previous / Next Cluster**.

## Related Repositories

- [smartsim-backend](https://github.com/SifeYassine/smartsim-backend): API version of SmartSim
- [smartsim-frontend](https://github.com/SifeYassine/smartsim-frontend): web interface

## Author

**Sifeddine Yassine** · [@SifeYassine](https://github.com/SifeYassine)
