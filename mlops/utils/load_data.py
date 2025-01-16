import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_csv_to_mongodb(csv_file, collection_name):
    """
    Load data from CSV into a MongoDB collection.
    """
    # MongoDB connection
    connection_string = os.getenv("MONGO_CONN_STRING")
    client = MongoClient(connection_string)
    db = client['fake_account_data']
    collection = db[collection_name]

    # Read CSV into DataFrame
    df = pd.read_csv(csv_file)

    # Insert records into MongoDB collection
    records = df.to_dict('records')
    collection.insert_many(records)
    print(f"Inserted {len(records)} records into {collection_name}")
