import base64
import os
from flask import Flask, request, jsonify
from flask_pymongo import pymongo
from dotenv import load_dotenv
import logging
from bson import json_util
import numpy as np
import cv2
import torch
from flask_cors import CORS

#loading the env variables from the .env file
load_dotenv()

#Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#Accessing the secret key from the env variables
mongo_pwd = os.getenv('MONGO_PASSWORD')
mongo_username = os.getenv('MONGO_USERNAME')

# Configure the logging format and level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

#Starting flask application
app = Flask(__name__)

#Enable CORS for all routes
CORS(app)

#URI to hit the request to our particular server
uri = f"mongodb+srv://{mongo_username}:{mongo_pwd}@cluster0.7nnu7au.mongodb.net/?retryWrites=true&w=majority"

#accessing Mongo Client
client = pymongo.MongoClient(uri)
db = client.get_database('computer-vision')
collection_name = 'images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = 'uploads'

#Method only allows certain extensions such as png, jpg, jpeg and gif to go through
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if collection_name not in db.list_collection_names():
    db.create_collection(collection_name)

collection = db[collection_name]

@app.route('/upload-image', methods=['POST'])
def upload_image():
    """
    Uploads an image along with its annotations to the MongoDB server

    Args:
        image: Image file in the req form data
        annotations: JSON object containing a list of dictionaries with bounding boxes and label names

    :return:
        JSON response indicating the success/failure of the upload operation

    """
    if 'image' not in request.files:
        return jsonify(message='Image not in request'), 400

    file = request.files['image']

    if file and allowed_file(file.filename):
        """
        Filename has been used as the primary key in the given project
        in the interest of time. It is not ideal, as multiple files can have 
        the same name. A possible workaround is to add the current timestamp to the file
        but for the sake of simplicity that has not been done here
        """
        if collection.find_one({'filename': file.filename}) is not None:
            return jsonify(message='File already exists'), 400

        image_data = file.read()

        if len(image_data) < 16777216: #or 16 mb
            encoded_image_data = base64.b64encode(image_data)

            filename = file.filename
            annotations = request.form.get('annotations')
            db_entry = {
                'image': encoded_image_data,
                'filename': filename,
                'annotations': annotations
            }

            collection.insert_one(db_entry)

            return jsonify(message='File uploaded successfully'), 200

        return jsonify(message='File too big'), 400

    return jsonify(message='Invalid request'), 400

@app.route('/delete-image/<filename>', methods=['DELETE'])
def delete_image(filename):
    """
    Deletes the image data and image annotation from the MongoDB server

    Args:
        filename: Image filename in params that needs to be deleted

    :return:
        JSON response indicating the success/failure of the delete operation

    """
    result = collection.delete_one({'filename': filename})

    if result.deleted_count == 1:
        return jsonify(message='File deleted successfully'), 200
    else:
        return jsonify(message='File not found in the database'), 404

@app.route('/get-image/<filename>', methods=['GET'])
def get_image(filename):
    """
    Fetches the image data and image annotation of a single document from the MongoDB server

    Args:
        filename: Image filename in params that needs to be fetched

    :return:
        JSON response containing the image, filename and annotations for the given image
        if successful and error response if not

    """
    db_entry = collection.find_one({'filename': filename})

    image_data_base64 = db_entry['image'].decode('utf-8')

    if db_entry:
        # Return image filename and annotations as JSON response
        response = {
            'filename': db_entry['filename'],
            'annotations': db_entry['annotations'],
            'image' : image_data_base64
        }
        return jsonify(response), 200
    else:
        return jsonify(message='File not found'), 404

@app.route('/get-all', methods=['GET'])
def get_all():
    """
    Fetches the image data and image annotation of all documents from the MongoDB server

    Args:
        N/A

    :return:
        JSON response with the objects of all images and annotations on the MongoDB server
        and error if not

    """

    all_documents = list(collection.find({}))

    # Convert binary image data to base64 encoding
    for doc in all_documents:
        if 'image' in doc:
            doc['image'] = doc['image'].decode('utf-8')

    # Serialize the documents to JSON using bson.json_util
    response = json_util.dumps(all_documents)
    return response, 200, {'Content-Type': 'application/json'}\


@app.route('/edit/<filename>', methods=['PUT'])
def edit_annotations(filename):
    """
        Updates the image annotation of a specific image from the MongoDB server

        Args:
            annotations and filename

        :return:
            JSON response indicating the success/failure of the update operation

        """
    new_annotations = request.json.get('annotations', '')

    # Update annotations for the given filename in the database
    result = collection.update_one({'filename': filename}, {'$set': {'annotations': new_annotations}})

    if result.modified_count == 1:
        return jsonify(message='Annotations updated successfully'), 200
    else:
        return jsonify(message='File not found or annotations not updated'), 404

@app.route('/get-predictions/<filename>', methods=['GET'])
def get_predictions(filename):
    """
        Gets the image filename and returns the predictions for bboxes and labels for the
        objects

        Args:
            filename

        :return:
            JSON response with the object containing the predictions if successful
            and error response if unsuccessful

        """
    db_entry = collection.find_one({'filename': filename})
    image_bytes = base64.b64decode(db_entry['image'])

    if not db_entry or not image_bytes:
        return jsonify(message='File not found'), 404

    nparr = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = model(img)

    filtered_predictions = []

    for pred in result.pred:
        labels = pred[:, -1].int()
        scores = pred[:, -2]
        boxes = pred[:, :-2]

        # Filter predictions with confidence above the threshold
        mask = scores > 0.5
        filtered_labels = labels[mask]
        filtered_scores = scores[mask]
        filtered_boxes = boxes[mask]

        # Create annotations for filtered predictions
        annotations = []
        for label, score, box in zip(filtered_labels, filtered_scores, filtered_boxes):
            x1, y1, x2, y2 = box.tolist()
            annotation = {
                "label": result.names[int(label)],
                "confidence": float(score),
                "bbox": [x1, y1, x2, y2]
            }
            annotations.append(annotation)

        filtered_predictions.append(annotations)

    return jsonify(filtered_predictions), 200


if __name__ == '__main__':
    app.run()
