import os
import joblib
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
from pymongo import MongoClient
from bson import ObjectId
import math
import requests
import re
import json
from bs4 import BeautifulSoup

os.chdir(r"models")

model_paths = {
    'gsr' : [r"gsr\kmeans_model_gsr.pkl",
             r"gsr\gmm_model_gsr.pkl",
             r"gsr\isolation_forest_gsr.pkl",
             r"gsr\scaler_gsr.pkl",
             r"gsr\autoencoder_gsr.pth",
             1],
    
    'imu' : [r"imu\kmeans_model_imu.pkl",
             r"imu\gmm_model_imu.pkl",
             r"imu\isolation_forest_imu.pkl",
             r"imu\scaler_imu.pkl",
             r"imu\autoencoder_imu.pth",
             6],
    
    'srt' : [r"srt\kmeans_model_srt.pkl",
             r"srt\gmm_model_srt.pkl",
             r"srt\isolation_forest_srt.pkl",
             r"srt\scaler_srt.pkl",
             r"srt\autoencoder_srt.pth",
             3],
    
    'ecg' : [r"ecg\kmeans_model_ecg.pkl",
             r"ecg\gmm_model_ecg.pkl",
             r"ecg\isolation_forest_ecg.pkl",
             r"ecg\scaler_ecg.pkl",
             r"ecg\autoencoder_ecg.pth",
             125]
}

def rate_anomaly(cluster, gmm, iso, scaler, autoenc, input_dim, new_movement):
    clustering_model = joblib.load(cluster)
    gmm_model = joblib.load(gmm)
    iso_forest = joblib.load(iso)
    scaler = joblib.load(scaler)
    
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8)
            )
            self.decoder = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, input_dim)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    #input_dim = 1
    autoencoder = Autoencoder(input_dim)
    autoencoder.load_state_dict(torch.load(autoenc))
    autoencoder.eval()

    def detect_anomalies(new_sample):
        """
        Given a new IMU data sample, detect if it's an anomaly.
        """
        new_sample = np.array(new_sample).reshape(1, -1)

        new_sample = scaler.transform(new_sample)

        cluster_label = clustering_model.predict(new_sample)[0]

        with torch.no_grad():
            new_tensor = torch.tensor(new_sample, dtype=torch.float32)
            reconstructed_sample = autoencoder(new_tensor).detach().numpy()

        reconstruction_error = np.mean((new_sample - reconstructed_sample) ** 2)

        isolation_score = iso_forest.decision_function(new_sample)

        recon_threshold = np.percentile(reconstruction_error, 95)
        print(reconstruction_error)
        iso_threshold = np.percentile(isolation_score, 5)

        is_reconstruction_anomaly = reconstruction_error > recon_threshold
        is_isolation_anomaly = isolation_score < iso_threshold
        is_anomaly = is_reconstruction_anomaly or is_isolation_anomaly

        return {
            "Anomaly Detected": is_anomaly,
            "Assigned Cluster": cluster_label,
            "Reconstruction Error": reconstruction_error,
            "Isolation Score": isolation_score[0]
        }

    result = detect_anomalies(new_movement)
    print("\nPrediction Result:", result)
    return result["Reconstruction Error"]


MONGO_URI = "mongodb://localhost:27017/"  
DATABASE_NAME = "StrokeSenseAI"
COLLECTION_NAME = "Personal"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

def update_document(movements):
    data = get_health_data()
    acc = math.sqrt(movements[0][0]**2+movements[0][1]**2+movements[0][2]**2)
    gyro = math.sqrt(movements[0][3]**2+movements[0][4]**2+movements[0][5]**2)
    data.accelerometer.pop()
    data.accelerometer.append(acc)
    data.gyroscope.pop()
    data.gyroscope.append(gyro)
    result = collection.update_one({"_id": ObjectId('67c8988e624603c3ae1c0606')}, {"$set": update_dict})
    return result.modified_count 


def get_health_data():
    # Fetch the document
    data = collection.find_one({'_id': ObjectId('67c8988e624603c3ae1c0606')})
    
    # Convert ObjectId to string
    if data:
        data['_id'] = str(data['_id'])
    
    return data

def delete_data(query):
    result = collection.delete_one(query)
    print(f"Deleted {result.deleted_count} document(s)")

def extract_floats_from_url(url):
    try:
        response = requests.get(url, timeout=5)  # Timeout added for stability
        response.raise_for_status()

        # Try JSON first (in case of API responses)
        try:
            data = response.json()
            return extract_floats_from_text(json.dumps(data))
        except ValueError:
            pass  # Not JSON, fallback to HTML scraping

        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ")  # Ensures proper spacing

        return extract_floats_from_text(text)
    
    except requests.exceptions.RequestException as e:
        print(f"Error processing {url}: {e}")
        return []

def extract_floats_from_text(text):
    # Improved regex for floats (handles negatives, decimals, and scientific notation)
    float_pattern = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
    return [float(match) for match in float_pattern.findall(text)]

def process_urls(url_list):
    all_floats = []
    for url in url_list:
        float_list = extract_floats_from_url(url)
        all_floats.append(float_list)
    
    return all_floats


def normalize(values):
    if len(values) < 6:
        raise ValueError("List must have at least 6 elements")

    # Normalize first 3 elements in range [-20, 20]
    min1, max1 = -20, 20
    norm1 = [(x - min1) / (max1 - min1) for x in values[:3]]

    # Normalize last 3 elements in range [-4000, 4000]
    min2, max2 = -4000, 4000
    norm2 = [(x - min2) / (max2 - min2) for x in values[-3:]]

    return norm1 + norm2

def emergency_call():
    account_sid = "<your_account_sid>"
    auth_token = "<your_auth_token>"
    twilio_phone_number = "<your_twilio_phone_number>"  
    recipient_phone_number = get_health_data()['emc_phno']

    client = Client(account_sid, auth_token)

    call = client.calls.create(
        to=recipient_phone_number,
        from_=twilio_phone_number,
        twiml="<Response><Say voice='alice' language='en-US'>Hello! This is your customized Twilio call. Have a great day!</Say></Response>"
    )

    print(f"Call initiated. Call SID: {call.sid}")

models = ['imu', 'srt']
weights = [1,1,1]
probability = 0
movements = process_urls(urls)
movements[1].insert(0,82)
movements[1].append(98)
update_document(movements)
count = 0
for sensor_models, wght, movement in zip(models, weights, movements):
    if sensor_models == 'imu':
        movement = normalize(movement)
    cluster = model_paths[sensor_models][0]
    gmm = model_paths[sensor_models][1]
    iso = model_paths[sensor_models][2]
    scaler = model_paths[sensor_models][3]
    autoenc = model_paths[sensor_models][4]
    input_dim = model_paths[sensor_models][5]
    print(input_dim)
    probability += wght*rate_anomaly(cluster, gmm, iso, scaler, autoenc, input_dim, movement)
    if probability > 1 : count+=1
    
    
if count > 1:
    print("""
---------
STROKE
---------
    """)
    emergency_call()
else:
    print("""
---------
SAFE
---------
    """)

