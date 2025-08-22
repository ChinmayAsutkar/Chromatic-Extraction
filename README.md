# Chromatic Extraction

A simple and easy-to-use tool for extracting and analyzing color information from images, with additional utilities for filtering, redacting, and processing images.

## 🚀 Easy to Use
- Just place your images in the project directory or specify the path.
- Run the provided Python scripts to extract colors, filter images, or redact subimages.
- No complex setup required—just install the dependencies and get started!

## 🛠️ Tech Stack Used
- **Python 3**
- **OpenCV**: For image processing and manipulation
- **NumPy**: For efficient numerical operations
- **scikit-image**: For advanced image analysis
- **Matplotlib**: For visualization (if needed)
- **Other Python libraries** as required (see `requirements.txt`)

## 🧠 Logic Used
- **Color Extraction**: Extracts dominant or specific colors from images using clustering and color space transformations.
- **Filtering**: Applies RANSAC and other filtering techniques to clean up or segment images.
- **Redaction**: Automatically detects and redacts sensitive or unwanted subimages.
- **Batch Processing**: Supports processing multiple images in folders for efficient workflows.

## 📂 Project Structure
- `color_extractor.py` – Main color extraction logic
- `filter_ransac.py` – Filtering and segmentation utilities
- `redact_subimages.py` – Redaction logic for subimages
- `singleimage.py` – Single image processing example
- `ui_filter_stamp.py` – UI for filtering and stamping
- `Archive/` – (Ignored) Storage for archived images
- `requirements.txt` – List of dependencies
- `test/` – Test scripts and utilities

## 🏁 Getting Started
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the desired script:
   ```bash
   python color_extractor.py
   ```

---

Feel free to contribute or open issues for improvements!
