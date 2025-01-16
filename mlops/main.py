from utils.load_data import load_csv_to_mongodb
import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from utils.fake_account_detector import FakeAccountDetector

# Load environment variables
load_dotenv()

# MongoDB connection
connection_string = os.getenv("MONGO_CONN_STRING")
client = MongoClient(connection_string)
db = client['fake_account_data']

# Collections
real_users_collection = db['real_users']
fake_users_collection = db['fake_users']

# Fetch data from MongoDB
real_users = pd.DataFrame(list(real_users_collection.find()))
fake_users = pd.DataFrame(list(fake_users_collection.find()))

# Drop '_id' column before saving to CSV
if "_id" in real_users.columns:
    data = real_users.drop(columns=["_id"])
    data.to_csv('mlops/data/users.csv', index=False)

if "_id" in fake_users.columns:
    data = fake_users.drop(columns=["_id"])
    data.to_csv('mlops/data/fake_users.csv', index=False)



# Initialize the detector
detector = FakeAccountDetector()

# Preprocess data and train the model
X, y = detector.load_and_preprocess_data('mlops/data/users.csv', 'mlops/data/fake_users.csv')
results = detector.train(X, y)

# Print the results
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# Plot feature importance and save the models
detector.plot_feature_importance()
detector.save_models('mlops/models')


load_csv_to_mongodb('mlops/data/users.csv', 'real_users')

load_csv_to_mongodb('mlops/data/fake_users.csv', 'fake_users')




