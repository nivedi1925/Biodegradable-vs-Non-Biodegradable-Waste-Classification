# Biodegradable-vs-Non-Biodegradable-Waste-Classification
A deep learning project that classifies images of waste into biodegradable and non-biodegradable categories using transfer learning with ResNet50. This is a computer vision project aimed at supporting waste segregation through intelligent image classification.

📌 Project Overview

    Goal: Automate classification of waste for better recycling and disposal.

    Model Used: ResNet50 (pretrained on ImageNet).

    Dataset: Non and Biodegradable Waste Dataset from Kaggle.

    Accuracy Achieved:

        ✅ Validation Accuracy: 98%

        ✅ Test Accuracy: 90%

📁 Dataset Details

    2 Classes: Biodegradable, Non-Biodegradable

    Total Images: 256K images (156K original data)

    Format: .jpg images in labeled folders

    Resized to 64x64

    Applied data augmentation (random flip, rotation)


Dataset Source : [Kaggle](https://www.kaggle.com/datasets/rayhanzamzamy/non-and-biodegradable-waste-dataset)<br>
Sample Images:
<img align="centre" alt="GIF" src="https://github.com/nivedi1925/Biodegradable-vs-Non-Biodegradable-Waste-Classification/blob/main/images/Screenshot%20from%202025-06-30%2000-03-53.png" />

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


Observations:

    Slight overfitting observed (val accuracy > test accuracy).

    Likely due to low number of epochs( Contraints here is limited computational capabilities).

    Potential improvements: more epochs, early stopping, regularization, fine-tuning more layers.



🧩 Future Work

    Fine-tune more layers in ResNet50

    Try other models like MobileNetV2 or EfficientNet

    Deploy model as a web app with Flask or Streamlit

    Use ExplanableAI tools such as LIME

🤝 Acknowledgments

    Dataset by contributors on Kaggle

    ResNet50 by Kaiming He et al.

    TensorFlow & Keras for model development

