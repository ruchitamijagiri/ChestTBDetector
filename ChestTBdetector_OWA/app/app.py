from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
import pydicom
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from flask import Flask
from flask_cors import CORS
from io import BytesIO
import io
import base64
from PIL import ImageEnhance

app = Flask(__name__, static_url_path='/static')
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://openmrs:6A|Wwr1|4jyQ@localhost:3317/openmrs'
db = SQLAlchemy(app)

#Set model path
MODEL_PATH = 'model/chexNet_30epoch_lr_001_latest.hd5'
BASEPATH = os.path.dirname(__file__)

model = load_model(MODEL_PATH)

class Person(db.Model):
    __tablename__ = 'person'

    person_id = db.Column(db.Integer, primary_key=True)
    gender = db.Column(db.String(10))
    birthdate = db.Column(db.Date)


class PersonName(db.Model):
    __tablename__ = 'person_name'

    person_name_id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('person.person_id'))
    given_name = db.Column(db.String(255))
    family_name = db.Column(db.String(255))


class Patient(db.Model):
    __tablename__ = 'patient'

    patient_id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('person.person_id'))

def load_and_preprocess_dicom_image(filename, target_size):
        dicom_data = pydicom.dcmread(filename)
        image = dicom_data.pixel_array.astype(np.float32) / np.max(dicom_data.pixel_array)
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.resize(target_size)
        image = np.array(image)
        rgb_image= make_rgb(image)
        rgb_image = np.expand_dims(rgb_image, axis=0)
        return rgb_image

def make_rgb(img):
        if img.ndim == 3:
            return img
        img3 = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        return img3

# Predict function
def model_predict( model, filenames):

    #Pre-process the image from original model code
    target_size = (256, 256)
    test_image =load_and_preprocess_dicom_image(filenames, target_size)

    # Make predictions (predict returns a tensor)
    preds = model.predict(test_image)
    print(preds)
    prediction= preds[:,1]
    print(float(prediction[0]))
    if float(1-prediction[0]) > float(prediction[0]):
        final_prediction='Signs of TB detected'
        score= float(1-prediction[0])
    else:
        final_prediction='Unlikely TB'
        score=float(prediction[0])

    return final_prediction, score #(1-preds)


#Define application route (For web app)
@app.route('/',methods=['GET'])
def index():
    print("Start index page")
    patient_id = request.args.get('patient_id')
    patient_name = request.args.get('patient_name')

    return render_template('index.html' , patient_id=patient_id, patient_name=patient_name)


@app.route('/upload-dicom', methods=['POST'])
def upload_dicom():
    dicom_file = request.files['dicomFile']
    dicom_image = pydicom.dcmread(dicom_file)
    max_size=(400, 400)

    # Convert the DICOM image to a PIL image
    pil_image = Image.fromarray(dicom_image.pixel_array)

    # Adjust the contrast and brightness
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(2.0)  # Increase contrast

    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(0.5)  # Reduce brightness

    # Convert the image to RGB mode if it is not already
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Resize the image while maintaining the aspect ratio
    pil_image.thumbnail(max_size)

    # Create a buffer to hold the image data
    image_buffer = io.BytesIO()

    # Save the PIL image as JPEG or PNG to the buffer
    image_format = 'JPEG' if pil_image.mode == 'RGB' else 'PNG'
    pil_image.save(image_buffer, format=image_format)

    # Get the base64-encoded image data
    image_data = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

    # Return the base64-encoded image data
    return jsonify({'image_data': image_data})

@app.route('/search', methods=['POST'])
def search_patients():
    search_query = request.form['searchQuery']
    
    # Perform the database query
    query = text("""
        SELECT p.identifier, pn.given_name, pn.family_name, pe.gender, TIMESTAMPDIFF(YEAR, pe.birthdate, CURDATE()) AS age
        FROM patient_identifier p
        JOIN person pe ON p.patient_id = pe.person_id
        JOIN person_name pn ON pe.person_id = pn.person_id
        WHERE pn.given_name LIKE CONCAT('%', :search_query, '%')
    """)
    
    # Execute the query with the search_query parameter
    results = db.session.execute(query, {'search_query': search_query})
    
    # Convert the query results to a list of dictionaries
    patients = []
    for row in results:
        patient = {
            'patient_id': row[0],
            'given_name': row[1],
            'family_name': row[2],
            'gender': row[3],
            'age': row[4]
        }
        patients.append(patient)
    
    return jsonify(patients)

@app.route('/patient-details/<patient_id>'  )
def patient_details(patient_id):
    # Perform the database query to retrieve the details of the selected patient
    query = text("""
        SELECT p.identifier, pn.given_name, pn.family_name, pe.gender, TIMESTAMPDIFF(YEAR, pe.birthdate, CURDATE()) AS age 
        FROM patient_identifier p 
        JOIN person pe ON p.patient_id = pe.person_id 
        JOIN person_name pn ON pe.person_id = pn.person_id 
        WHERE p.identifier = :patient_id
    """)
    
    query2=text(""" SELECT pa.address1, pa.address2, pa.city_village AS city, pa.state_province AS state, pa.country
        FROM person_address pa
        JOIN person pe ON pa.person_id = pe.person_id
        JOIN patient p ON pe.person_id = p.patient_id
        JOIN patient_identifier pi ON pi.patient_id= p.patient_id
        WHERE pi.identifier = :patient_id
        """)
    
    query3 = text(""" SELECT e.encounter_id, e.patient_id, e.encounter_datetime, et.name AS encounter_type_name
                FROM encounter e
                JOIN patient p ON e.patient_id = p.patient_id
                JOIN person pe ON e.patient_id = pe.person_id
                JOIN patient_identifier pi ON pi.patient_id = p.patient_id
                JOIN encounter_type et ON e.encounter_type = et.encounter_type_id
                WHERE pi.identifier = :patient_id
            """)
    
    patient=None
    address=None
    visit=[]
    
    # Execute the query with the patient_id parameter
    results = db.session.execute(query, {'patient_id': patient_id})
    
    # Fetch the first row of the query result
    row = results.fetchone()

    results2 = db.session.execute(query2, {'patient_id': patient_id})
    
    # Fetch the first row of the query result
    row2 = results2.fetchone()
    
    
    results3 = db.session.execute(query3, {'patient_id': patient_id})
    
    # Fetch all rows of the query result
    row3 = results3.fetchall()

    # Create a patient dictionary with the details of the selected patient
    if row:
        patient = {
        'patient_id': row[0],
        'given_name': row[1],
        'family_name': row[2],
        'gender': row[3],
        'age': row[4]
        }

    if row2:
        address = {
        'address1': row2[0],
        'city': row2[2],
        'state': row2[3],
        'country': row2[4]
        }

    print(row3)
    if row3:
        # Convert the query results to a list of dictionaries
        visit = [
            {
            'datetime': rows[2],
            'visit_type': rows[3]
        }
        for rows in row3
        ]
    print(visit)
    
    return render_template('patientDetails.html', patient=patient, address=address, visits=visit)

@app.route('/get-person-id', methods=['POST'])
def redirect_patient_page():
    patient_identifier = request.form['patient_identifier']

    # Get the patient ID associated with the patient identifier
    query=text(""" SELECT p.patient_id FROM patient p
                JOIN person pe ON p.patient_id = pe.person_id
                JOIN person_name pn ON pe.person_id = pn.person_id
                JOIN patient_identifier pi ON pi.patient_id = p.patient_id
                WHERE pi.identifier = :patient_identifier;
            """)
    
    results = db.session.execute(query, {'patient_identifier': patient_identifier})
    patient_id = results.fetchone()[0]

    response = {'patient_id': patient_id}
    print(response)

    # Return the response as JSON
    return response

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    imagedata = request.files['imagedata']

    # Check if post request has a file
    if 'imagedata' not in request.files:
        resp = jsonify({'message': 'No file in the request'})
        resp.status_code = 404  # Check status (code when "no data available")
        return resp

    # Store image to path ./uploads
    file_name = secure_filename(imagedata.filename)
    image_path = os.path.join(BASEPATH, 'uploads', 'testing', file_name)
    imagedata.save(image_path)

    # Call predictions function, return probabilities of positive and negative cases
    pred, score = model_predict(model, image_path)

    # Return the prediction result as JSON
    return jsonify({'prediction': str(pred), 'score': str(score)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

