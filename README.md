# Identification System using Face Recognition

This project is a Face Recognition System built with **Flask** as the backend framework, **TensorFlow** and **Keras** for training a Convolutional Neural Network (CNN) model, **OpenCV** for image preprocessing, and **face_recognition** library for face embedding extraction. The system allows you to upload face images, extract their embeddings, train a CNN model for face recognition, and make predictions on new images.

## Technologies Used

This project leverages a combination of various technologies to provide a robust and scalable face recognition system.

### 1. **Flask**
- Flask is used to build the web application API. It serves as the backend framework to handle requests, such as uploading images, training the model, and making predictions. Flask allows for easy API creation and supports serving the machine learning model via HTTP requests.

### 2. **TensorFlow & Keras**
- **TensorFlow** is used for building and training the deep learning model for face recognition. **Keras**, which is an API in TensorFlow, simplifies the process of building neural network models. The model is a Convolutional Neural Network (CNN) that takes face embeddings as input to classify individuals.

### 3. **face_recognition**
- **face_recognition** is a Python library built on **dlib**. It is used for face detection and extracting face embeddings. Each person in the system is represented by a 128-dimensional vector (embedding) derived from their facial features.

### 4. **OpenCV**
- **OpenCV** is used for general image preprocessing tasks such as resizing images and handling file uploads. It provides tools for manipulating images before passing them into the face recognition model.

### 5. **NumPy & scikit-learn**
- **NumPy** is used for handling numerical operations, such as reshaping and manipulating arrays.
- **scikit-learn** is used for data splitting and label encoding (converting person names to numeric labels).

## How It Works

### Step 1: Uploading Images
When an image is uploaded to the system, it is passed to the `face_recognition` library, which detects any faces in the image and extracts their embeddings. These embeddings are 128-dimensional vectors that uniquely represent the person's face.

### Step 2: Storing Embeddings
The face embeddings, along with the corresponding person's name (or label), are stored in a database or folder structure for later use in model training.

### Step 3: Model Training
The stored face embeddings are used to train a Convolutional Neural Network (CNN) model. This model is responsible for classifying which person is in a given image. The embeddings are used as input to the model, and the model learns to associate them with specific labels (person's name).

### Step 4: Making Predictions
Once the model is trained, it can be used to predict the identity of a person from a new image. When a new image is uploaded, the face embeddings are extracted, and the trained model classifies the image based on the embeddings.

### Training the Model

The model is trained using **face embeddings** extracted from images. These embeddings are 128-dimensional vectors representing each person's unique facial features. The training process involves the following steps:

1. **Extract face embeddings**: Using the `face_recognition` library, the embeddings for faces in each image are extracted and stored.
2. **Train a CNN model**: The embeddings are used as input features for training a Convolutional Neural Network (CNN). The CNN model is trained to classify these embeddings into different labels (people's names).
3. **Save the trained model**: After training, the model is saved and can be used to make predictions for new images.
   
### Model Architecture
The CNN model consists of:
- **Convolutional layers**: These layers help extract spatial features from the face embeddings.
- **MaxPooling layers**: Used to reduce the spatial dimensions of the feature maps.
- **Dropout layers**: Used to prevent overfitting during training by randomly setting some weights to zero.
- **Dense layers**: Fully connected layers for classification, which output the predicted label (person's name).

### Step 5: Model Storage
The trained model is saved in a file (e.g., `model.h5`) and can be reloaded for future predictions. The embeddings and labels are also saved to a file for further reference.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/AnT2003/Identification-System-using-Face-Recognition.git
    cd Identification-System-using-Face-Recognition
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Start the Flask Application:**
    Run the following command to start the Flask server:
    ```bash
    python app.py
    ```
    The server will be running at `http://127.0.0.1:8000/`.

2. **Training the Model:**
    To train the model, you need to have images of faces. Each image should contain only one face, and you should label the image with the person's name.
    
    Example of training:
    ```bash
    curl -X POST http://127.0.0.1:8000/train_model
    ```

    This will train the model using the face embeddings stored in the `uploads` folder. The trained model will be saved for future use.
- Using on flask web:

  <img width="944" alt="image" src="https://github.com/user-attachments/assets/6a6a056f-7dee-400b-9445-ac4b876fb2d1" />


3. **Upload an Image for Prediction:**
    To predict the identity of a person from an uploaded image, use the `predict_face` endpoint.
    
    Example of prediction:
    ```bash
    curl -X POST -F "file=@path_to_image.jpg" http://127.0.0.1:5000/predict_face
    ```

    The server will return the predicted name of the person in the image.
   
- Usung on flask web:

  <img width="959" alt="image" src="https://github.com/user-attachments/assets/05eca801-0c92-4f78-9a6b-1b698fa0159e" />


### Folder Structure
```
face-recognition-flask/
│
├── app.py                # Flask application
├── model/                # Directory for storing the trained model
│   └── model.h5          # The trained CNN model
├── uploads/              # Directory for storing uploaded images and embeddings
├── requirements.txt      # Python dependencies
└── README.md             # This file
```
