import os
import numpy as np
import face_recognition
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
import pickle
from flask import Flask, render_template, request, jsonify

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__)

# Define the main directory for saving uploaded files and models
BASE_UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'model'  # Folder where the model and label encoder are saved
if not os.path.exists(BASE_UPLOAD_FOLDER):
    os.makedirs(BASE_UPLOAD_FOLDER)
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

def save_images_and_folder(images, person_name, person_id):
    person_folder = os.path.join(BASE_UPLOAD_FOLDER, f"{person_name}_{person_id}")
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    
    for image_file in images:
        image_path = os.path.join(person_folder, image_file.filename)
        image_file.save(image_path)


def extract_face_embeddings(data_dir):
    known_face_encodings = []
    known_face_names = []
    known_face_ids = []

    for folder_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, folder_name)
        if not os.path.isdir(person_dir):
            continue

        # Ensure folder_name contains a '_'
        if '_' in folder_name:
            person_name, person_id = folder_name.rsplit('_', 1)
        else:
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(person_name)
                known_face_ids.append(person_id)
    return np.array(known_face_encodings), known_face_names, known_face_ids


def train_cnn_model(embeddings, labels):
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels_encoded, test_size=0.2, random_state=42)

    # Reshape the data for CNN input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Define the CNN model with adjusted hyperparameters and techniques
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.5),  # Reduced dropout rate to avoid excessive regularization
        Flatten(),
        Dense(len(label_encoder.classes_), activation='relu'),
        Dropout(0.5), 
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    # Compile the model with modified loss function
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model with early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), batch_size=16, callbacks=[early_stopping])

    # Extract the best epoch from the history
    best_epoch = np.argmin(history.history['val_loss'])
    print(f"Best epoch based on validation loss: {best_epoch + 1}")

    # Evaluate the model at the best epoch
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"CNN Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, label_encoder

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    if 'name' not in request.form or 'id' not in request.form or 'images' not in request.files:
        return jsonify({"error": "No name, id, or images provided"}), 400

    person_name = request.form['name']
    person_id = request.form['id']
    image_files = request.files.getlist('images')

    # Save images to the corresponding folder
    save_images_and_folder(image_files, person_name, person_id)

    # Extract face embeddings from all folders in the base upload folder
    embeddings, names, ids = extract_face_embeddings(BASE_UPLOAD_FOLDER)
    
    if len(embeddings) == 0 or len(names) == 0 or len(ids) == 0:
        return jsonify({"error": "No data found in the provided directory"}), 400

    labels = [f"{name}_{id}" for name, id in zip(names, ids)]
    cnn_model, label_encoder = train_cnn_model(embeddings, labels)

    # Save the model and label encoder in the designated model folder
    cnn_model.save(os.path.join(MODEL_FOLDER, 'cnn_face_recognizer_model.h5'))
    with open(os.path.join(MODEL_FOLDER, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    return jsonify({"message": "Model trained and saved successfully"}), 200

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    # Save the uploaded image
    image_path = os.path.join(BASE_UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    # Load the model and label encoder from the default model folder
    cnn_model_path = os.path.join(MODEL_FOLDER, 'cnn_face_recognizer_model.h5')
    label_encoder_path = os.path.join(MODEL_FOLDER, 'label_encoder.pkl')

    if not os.path.exists(cnn_model_path) or not os.path.exists(label_encoder_path):
        return jsonify({"error": "Model or label encoder not found in the model folder"}), 400

    cnn_model = tf.keras.models.load_model(cnn_model_path)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    # Predict
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if len(face_encodings) == 0:
        return jsonify({"error": "No faces found in the image"}), 400

    predictions = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_encoding = face_encoding.reshape((1, face_encoding.shape[0], 1))
        prediction = cnn_model.predict(face_encoding)
        predicted_label_index = np.argmax(prediction)
        predicted_label = label_encoder.classes_[predicted_label_index]
        accuracy = np.max(prediction) * 100

        predicted_name, predicted_id = predicted_label.rsplit('_', 1)
        predictions.append({
            "name": predicted_name,
            "id": predicted_id,
            "accuracy": accuracy,
            "bounding_box": {
                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left
            }
        })
    return jsonify({"predictions": predictions}), 200


if __name__ == '__main__':
    app.run(debug=True, port=8000)
