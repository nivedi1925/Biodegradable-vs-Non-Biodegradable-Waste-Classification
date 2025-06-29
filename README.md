# Biodegradable-vs-Non-Biodegradable-Waste-Classification
A deep learning project that classifies images of waste into biodegradable and non-biodegradable categories using transfer learning with ResNet50. This is a computer vision project aimed at supporting waste segregation through intelligent image classification.
Project Overview
📌 Project Overview

    Goal: Automate classification of waste for better recycling and disposal.

    Model Used: ResNet50 (pretrained on ImageNet).

    Dataset: Non and Biodegradable Waste Dataset from Kaggle.

    Accuracy Achieved:

        ✅ Validation Accuracy: 98%

        ✅ Test Accuracy: 90%

📁 Dataset Details

    2 Classes: Biodegradable, Non-Biodegradable

    Total Images: ~2,500+

    Format: .jpg images in labeled folders

    Preprocessing:

        Resized to 224x224

        Normalized to [0, 1]

        Applied data augmentation (random flip, rotation)

🧠 Model Architecture

    Base: ResNet50 (frozen base layers)

    Added layers:

        Global Average Pooling

        Dense (ReLU)

        Dropout (optional)

        Dense (Softmax for 2-class classification)
🏋️ Training Details

    Epochs: 4

    Optimizer: Adam

    Loss Function: Categorical Crossentropy

    Batch Size: 32

    Learning Rate: 0.0001

📊 Results & Evaluation
Metric	Validation	Test
Accuracy	98%	90%
Loss	~0.05	~0.25
Confusion Matrix (Test Set)

(Add an image or code snippet if available)
Observations:

    Slight overfitting observed (val accuracy > test accuracy).

    Likely due to limited dataset size and low number of epochs.

    Potential improvements: more epochs, early stopping, regularization, fine-tuning more layers.

🚀 How to Run

    Clone the repo:

git clone https://github.com/yourusername/biodegradable-classification.git
cd biodegradable-classification

Install requirements:

pip install -r requirements.txt

Run training:

python train.py

Evaluate model:

    python evaluate.py

🧩 Future Work

    Fine-tune more layers in ResNet50

    Try other models like MobileNetV2 or EfficientNet

    Deploy model as a web app with Flask or Streamlit

    Use Grad-CAM for explainability

🤝 Acknowledgments

    Dataset by contributors on Kaggle

    ResNet50 by Kaiming He et al.

    TensorFlow & Keras for model development

