Patient Data Analysis and Classification

This project provides a comprehensive pipeline for processing patient data, combining numeric features with image data, and building a classification model utilizing deep learning and attention mechanisms. The project includes preprocessing, data integration, feature extraction, and model training to achieve an efficient and interpretable workflow.

Project Overview

The project consists of four main steps:
	1.	Patient Data Processing (1_Patient_data_process.ipynb)
Process raw patient data to extract key numeric features, clean data inconsistencies, and prepare the dataset for downstream tasks.
	2.	Image Data Integration (2_combine_patient_img.ipynb)
Combine patient numeric features with corresponding image data. Align and format the data to facilitate multi-modal learning.
	3.	Feature Extraction and Dataset Preparation (3.1.code_to_pic.ipynb)
Convert raw hex image data into standardized image formats. Generate combined image features and corresponding numeric features for downstream classification tasks.
	4.	Model Training with Attention (3.2_pytouch_attention.ipynb)
Build a multi-modal classification model that integrates numeric and image features with attention mechanisms to focus on relevant information.

Pipeline Details

1. Patient Data Processing

	•	Input: Raw patient data in Excel files.
	•	Steps:
	•	Filter relevant columns and clean missing values.
	•	Pivot the dataset to organize numeric features such as ALB, ALT, AST, GLO, WBC, and TP.
	•	Fill missing numeric values based on temporal proximity within the same patient.
	•	Output: A clean and processed CSV file with numeric patient features (processed_patient_data.csv).

2. Image Data Integration

	•	Input: Numeric features from Step 1 and image data in hexadecimal format.
	•	Steps:
	•	Convert hexadecimal image data into raw images.
	•	Pivot image data to organize multiple images (TX0, TX1, etc.) per patient.
	•	Merge numeric features with image data based on patient ID and date.
	•	Output: A combined dataset (all_data.csv) with aligned numeric and image features.

3. Feature Extraction and Dataset Preparation

	•	Input: Combined dataset (all_data.csv) from Step 2.
	•	Steps:
	•	Convert hexadecimal image data into grayscale images of size (150, 150).
	•	Horizontally concatenate multiple images for each patient into a single composite image.
	•	Save numeric features and composite images for each patient into category-based folders.
	•	Output: A structured dataset directory:

dataset/
├── category_1/
│   ├── patient_1/
│   │   ├── combined_image.jpg
│   │   └── features.csv
│   ├── patient_2/
│   │   ├── ...
└── ...



4. Model Training with Attention

	•	Input: Preprocessed dataset from Step 3.
	•	Steps:
	•	Load numeric and image features using PyTorch’s DataLoader.
	•	Construct a multi-modal classification model:
	•	Image features are extracted using a ResNet50 model.
	•	Numeric features are concatenated with attention-enhanced image features.
	•	A fully connected classifier predicts patient categories.
	•	Train the model and evaluate performance using metrics like accuracy, precision, recall, and F1-score.
	•	Output: Trained model weights and performance metrics.

Model Performance

Metric	Value
Accuracy	86.5%
Precision	77.4%
Recall	86.5%
F1-Score	0.8109

Usage Instructions

	1.	Clone the repository:

git clone <repository-url>
cd <repository-folder>


	2.	Install Dependencies:
Ensure you have Python 3.9+ and the required libraries:

pip install -r requirements.txt


	3.	Prepare the Dataset:
	•	Place raw patient data (.xls files) and image data in the raw_data/ folder.
	•	Update file paths in the notebooks if necessary.
	4.	Run the Pipeline:
Execute the notebooks in the following order:
	•	1_Patient_data_process.ipynb
	•	2_combine_patient_img.ipynb
	•	3.1.code_to_pic.ipynb
	•	3.2_pytouch_attention.ipynb
	5.	Train the Model:
The model will be trained with early stopping to prevent overfitting. Results will be logged and saved.

Key Features

	•	Multi-Modal Learning: Combines numeric and image features for patient classification.
	•	Attention Mechanism: Enhances the focus on relevant image regions.
	•	Comprehensive Pipeline: Includes data preprocessing, integration, and training.
	•	Customizable: Easily adjust parameters like image size, model architecture, and training settings.

Future Enhancements

	•	Expand the dataset to include additional patient data and modalities.
	•	Explore advanced attention mechanisms for better interpretability.
	•	Optimize hyperparameters to improve classification performance.

Feel free to contribute to this project by submitting issues or pull requests!
