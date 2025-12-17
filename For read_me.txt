# Crack Tracking and Analysis using U-Net Segmentation

This Google Colab notebook implements a deep learning pipeline for tracking and analyzing cracks in materials from image sequences. It uses a U-Net model for semantic segmentation of crack regions, followed by advanced image processing to extract quantitative measurements like crack length, crack tip opening displacement (CTOD), tip radius, and crack propagation.

## Table of Contents
1.  [Overview](#1-overview)
2.  [Features](#2-features)
3.  [Setup and Prerequisites](#3-setup-and-prerequisites)
4.  [How to Run](#4-how-to-run)
5.  [Notebook Structure and Key Steps](#5-notebook-structure-and-key-steps)
6.  [Outputs and Results](#6-outputs-and-results)
7.  [Interactive Elements and GitHub Display](#7-interactive-elements-and-github-display)

--- 

## 1. Overview

This project focuses on analyzing crack propagation in a time-series dataset. A U-Net model is trained (or a pre-trained one is used) to accurately segment crack regions. Subsequently, custom functions extract critical fracture mechanics parameters from the segmented masks, providing insights into crack growth dynamics and blunting behavior over time.

## 2. Features

*   **U-Net based Semantic Segmentation**: Automated crack detection and segmentation.
*   **Google Drive Integration**: Seamlessly loads data from and saves results to Google Drive.
*   **Comprehensive Crack Measurement**: Calculates:
    *   Crack tip position (leftmost point, assuming leftward propagation)
    *   Crack length
    *   Crack area
    *   Crack Tip Opening Displacement (CTOD) at various distances from the tip
    *   Maximum crack width
    *   Crack tip radius (bluntness)
*   **Nanometer-scale Analysis**: All measurements are converted from pixels to nanometers (1 pixel = 0.4 nm).
*   **Temporal Analysis Plots**: Visualizations of crack growth, CTOD, tip blunting, and area over time.
*   **Key Events Analysis**: Identifies significant crack growth, maximum blunting, and maximum CTOD.
*   **Visual Montage**: Generates an image montage showing crack evolution with overlays and measurements.

## 3. Setup and Prerequisites

This notebook is designed to run in [Google Colab](https://colab.research.google.com/).

1.  **Google Account**: You'll need a Google account to access Colab and Google Drive.
2.  **Google Drive Folder Structure**: Ensure your Google Drive is set up with the following directory structure:
    ```
    MyDrive/
    └── Crack_tracking/
        ├── data/
        │   └── 1508 20250613 105 kx Ceta Camera.npy  # Your raw image data
        └── annotations/
            ├── 1508_20250613_105_kx_Ceta_Camera_annotated_masks.npy
            └── 1508_20250613_105_kx_Ceta_Camera_frame_indices.npy
    ```
    The notebook will create a `results/` folder inside `Crack_tracking/` to store outputs.
3.  **Python Packages**: The notebook will install all necessary packages automatically (`torch`, `torchvision`, `segmentation-models-pytorch`, `albumentations`, `tqdm`).

## 4. How to Run

1.  **Open in Google Colab**: Click the "Open in Colab" badge or upload the `.ipynb` file to your Google Colab environment.
2.  **Mount Google Drive**: The first code cell will prompt you to connect to your Google Drive. Authorize access so the notebook can read data and save results.
3.  **Execute Cells**: Run all cells in the notebook sequentially. You can do this by selecting `Runtime -> Run all` from the Colab menu. The notebook includes checks to ensure variables are defined even if the runtime restarts, enhancing reproducibility.

## 5. Notebook Structure and Key Steps

The notebook follows a logical flow:

*   **Data Loading & Preparation**: Loads raw image data and annotated masks from Google Drive. Performs train/validation/test splits.
*   **Dataset & Dataloaders**: Defines `CrackDataset` for data handling and `DataLoader`s for efficient batch processing, including data augmentations.
*   **Model Definition & Loss**: Sets up the U-Net segmentation model, loss function (DiceBCELoss), and optimizer.
*   **Model Training (Optional)**: If the `_best_model.pth` is not found, the model will be trained. Otherwise, the best saved model is loaded. **(This is the most computationally intensive step, but can be skipped if a trained model is available on Drive).**
*   **Inference & Segmentation**: Applies the (trained) U-Net model to all frames to generate binary crack masks.
*   **Measurement Extraction**: Uses advanced image processing to extract quantitative crack parameters, converting all to nanometers.
*   **Temporal Analysis**: Plots key crack measurements over time.
*   **Key Events Analysis**: Summarizes significant milestones in crack growth and blunting.
*   **Visualizations**: Generates image montages with crack overlays and measurements.

## 6. Outputs and Results

All generated outputs are saved to the `Crack_tracking/results/` folder on your Google Drive:

*   `{sample_name}_all_predictions.npy`: A NumPy array containing the binary segmentation masks for all frames.
*   `{sample_name}_measurements.csv`: A CSV file with all extracted crack measurements (length, CTOD, tip radius, etc.) in nanometers for each frame.
*   `{sample_name}_temporal_analysis.png`: Plots showing the evolution of crack parameters over time.
*   `{sample_name}_evolution_montage.png`: A montage of selected frames with crack overlays and measurements.
*   `{sample_name}_best_model.pth`: The saved weights of the best-performing U-Net model.

## 7. Interactive Elements and GitHub Display

This notebook contains interactive elements (e.g., `tqdm` progress bars, `matplotlib` plots displayed directly in the output cells) that are native to the Google Colab environment. 

*   **GitHub Rendering**: When viewed directly on GitHub, these interactive elements and dynamic outputs (like the `matplotlib` plots) may not render correctly or appear as static images without full interactivity. 
*   **Recommended Viewing**: For the best experience, including interactive plots and full execution capabilities, it is highly recommended to **open and run this notebook directly in Google Colab.**
*   **Static Rendering Alternatives**: If you need to share a static view with rendered outputs, consider using services like [nbviewer](https://nbviewer.jupyter.org/) by pasting the notebook's GitHub URL there. This will render the notebook with its saved outputs as static images, but you will still lose interactivity. 

To batch run this notebook without manual interaction, you could explore Google Colab's programmatic execution options (e.g., using the Colab API or tools like `papermill`), though this would require additional setup beyond the scope of this README.