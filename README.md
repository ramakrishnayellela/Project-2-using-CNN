# Plant Disease Detection using Computer Vision and Deep Learning

## 🌱 Project Overview

##### This project aims to identify plant diseases using computer-assisted aggregation techniques. It leverages Deep Learning (CNNs) to classify plant diseases based on leaf images and recommends suitable pesticides. The model is trained on a dataset containing images of healthy and diseased plant leaves.

## 🚀 Features

#### Image-based plant disease classification using Convolutional Neural Networks (CNNs)

#### Custom-trained model (not using pre-trained models)

#### Deployment-ready using Django for web or mobile integration

#### Disease diagnosis with pesticide recommendations

#### User-friendly GUI or web interface for real-time predictions

## 📂 Dataset

#### The dataset consists of labeled images of healthy and diseased leaves.

#### Sources: Publicly available datasets such as PlantVillage.

#### Image preprocessing includes resizing, normalization, and augmentation.

## 🛠️ Tech Stack

#### Programming Language: Python

#### Deep Learning Frameworks: TensorFlow / PyTorch

#### Data Handling: OpenCV, NumPy, Pandas

#### Deployment: Django

#### Visualization: Matplotlib, Seaborn

## 📌 Installation

### Clone the repository:

git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection

### Install dependencies:

pip install -r requirements.txt

Download the dataset and place it in the data/ directory.

Train the model:

python train.py --epochs 20 --batch_size 32 --model custom

Run the application:

python manage.py runserver

Open your browser and go to http://127.0.0.1:8000/.

## 🏆 Model Performance

#### Achieved accuracy: ~95% on the validation dataset.

Evaluated using Precision, Recall, F1-score, and Confusion Matrix.

Fine-tuned using Data Augmentation & Transfer Learning.



## 🖼️ Usage

Upload an image of a plant leaf.

The model predicts the disease category.

Displays the recommended pesticide for treatment.

### 🛠 Future Improvements

Increase dataset size for better generalization.

Deploy as a mobile app for farmers.

Improve explainability using Grad-CAM visualization.

### 📜 License

This project is open-source under the MIT License.

### 🤝 Acknowledgments

PlantVillage Dataset

TensorFlow/PyTorch Community

Deep Learning Enthusiasts & Researchers

✨ Contributions Welcome! 🚀

Feel free to contribute by improving the model, optimizing the web app, or adding new features.
