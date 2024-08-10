from pymongo import MongoClient
from gridfs import GridFS
import os
from dotenv import load_dotenv

# Set up MongoDB client and GridFS
mongo_client = MongoClient(os.getenv('MONGO_URI'))
db = mongo_client['image_database']
collection = db['floor_maps']