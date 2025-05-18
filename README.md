Certainly! Here's a comprehensive `README.md` for the [Multimodal-Visual-Question-Answering](https://github.com/subham-agarwal05/Multimodal-Visual-Question-Answering/tree/main) repository, with a particular emphasis on the `inference.py` script:

---

# Multimodal Visual Question Answering

This project implements a Visual Question Answering (VQA) system that leverages multimodal data—combining visual and textual information—to answer questions about images. It integrates advanced transformer-based models for both image and text processing, facilitating accurate and context-aware responses.

## 📁 Repository Structure

* `inference.py` – Core script for performing inference using the trained VQA model.
* `curation.ipynb` – Notebook for data curation and preprocessing tasks.
* `requirements.txt` – Lists all Python dependencies required to run the project.
* `merged_finetuning/` – Directory containing fine-tuned model checkpoints.
* `BaselineEvaluations/` – Contains evaluation scripts and results for baseline models.([GitHub][1])

## 🧠 Model Overview

We employ BLIP, BLIP2-OPT, ViltBERT and Matcha. The baseline evaluation results for these can be found in the `BaselineEvaluations/` folder. We also carry out data curation with Gemini (flash1.5) on the Amazon Berkeley Objects (ABO) dataset to create the dataset. This was used to fine-tune the dataset.
BLIP (Bootstrapped Language-Image Pretraining) serves as the foundation for general-purpose VQA. Its encoder-decoder architecture is pre-trained on large-scale image-text pairs, allowing it to effectively interpret questions and produce fluent answers grounded in natural images. It's particularly useful in cases requiring object recognition or contextual understanding in everyday photographs.

BLIP-2, an advancement over the original BLIP architecture, introduces a vision-to-language bridge that connects frozen vision encoders with large-scale language models like OPT-2.7B. This configuration enables more sophisticated reasoning and open-ended generation capabilities. By leveraging the scale and expressiveness of the OPT model, BLIP-2 can handle complex, free-form queries that require deeper understanding, making it suitable for tasks where nuanced language modeling is critical.

Matcha, a specialized variant of Google’s Pix2Struct architecture, is incorporated to handle domain-specific VQA on structured data visualizations. This model is fine-tuned on the ChartQA dataset and excels at reading charts, plots, and graphs. It transforms images into grid-based token sequences and processes them with a text decoder, making it highly effective for extracting quantitative data and answering questions that involve interpreting visualized statistics or trends.

## 🚀 Inference Guide

### Prerequisites

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/subham-agarwal05/Multimodal-Visual-Question-Answering.git
   cd Multimodal-Visual-Question-Answering
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models**:

   Ensure that the `merged_finetuning/` directory contains the necessary fine-tuned model checkpoints. If not, download them from the provided sources or train the model as per the instructions.

### Running Inference

The `inference.py` script is designed to answer questions based on input images. It processes the image and question, feeds them through the multimodal model, and outputs the predicted answer.([GitHub][1], [GitHub][2])

**Usage**:

```bash
python inference.py ----image_dir <path_to_images_folder> --csv_path "<csv file containing "question", "answer", "image_name">"
```

**Output**:

The script will generate the answers in results.csv

### Notes

* Ensure that the image file exists at the specified path and is in a supported format (e.g., JPG, PNG).
* The question should be a clear and concise natural language query related to the image content.([GitHub][2])

## 📊 Evaluation

The `BaselineEvaluations/` directory contains scripts and results for evaluating the performance of baseline models on standard VQA datasets. These evaluations help in benchmarking the effectiveness of the implemented model.
