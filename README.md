# Biodegradable-vs-Non-Biodegradable-Waste-Classification

📌 Project Overview

This computer vision project aims to automate waste segregation by classifying images of waste into biodegradable and non-biodegradable categories. Leveraging the power of deep learning and transfer learning, the project utilizes the ResNet50 architecture to perform image classification with high accuracy. The final model is deployed as a web application to demonstrate real-time predictions.

Dataset Source : [Kaggle](https://www.kaggle.com/datasets/rayhanzamzamy/non-and-biodegradable-waste-dataset)<br>
Sample Images:
<img align="centre" alt="GIF" src="https://github.com/nivedi1925/Biodegradable-vs-Non-Biodegradable-Waste-Classification/blob/main/images/Screenshot%20from%202025-06-30%2000-03-53.png" />

🎯 Objective

To build an intelligent system that:
- Helps in automated waste sorting.
- Supports environmental sustainability through smart waste management.
- Provides a user-friendly web interface for image-based classification

    
🔍 Key Features

- Transfer Learning with ResNet50: Utilizes pretrained ResNet50 for feature extraction and fine-tuning for binary classification.
- Image Preprocessing: Includes resizing, normalization, and data augmentation to improve model generalization.
- Web App Deployment: Simple UI built with Streamlit for uploading waste images and getting real-time predictions.

📁 Dataset Details

    2 Classes: Biodegradable, Non-Biodegradable

    Total Images: 256K images (156K original data)

    Format: .jpg images in labeled folders

    Resized to 64x64

    Applied data augmentation (random flip, rotation)


🧰 Tech Stack

    Language: Python

    Deep Learning Framework: TensorFlow / Keras

    Model: ResNet50 (Transfer Learning)

    Web Framework: Streamlit 

    Others:  NumPy, Matplotlib, Pandas


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
    
    Loss	~0.02	~0.30


🔎  Observations:

- Slight overfitting observed (val accuracy > test accuracy).
- Likely due to low number of epochs( Contraints here is limited computational capabilities).
- Potential improvements: more epochs, early stopping, regularization, fine-tuning more layers.

🎥 Screenshots of Web application:


![](images/ezgif.com-video-to-gif-converter(1).gif)

![](images/bio1.png)  ![](images/bio2.png)





🧩 Future Work

    Fine-tune more layers in ResNet50

    Try other models like MobileNetV2 or EfficientNet

    Deploy model as a web app with Flask or Streamlit

    Use ExplanableAI tools such as LIME

🤝 Acknowledgments

    Dataset by contributors on Kaggle

    ResNet50 by Kaiming He et al.

    TensorFlow & Keras for model development

