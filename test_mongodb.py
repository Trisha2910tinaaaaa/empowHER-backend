import os
from dotenv import load_dotenv
from pymongo import MongoClient
import datetime
import uuid

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

def test_mongodb_connection():
    """Test connection to MongoDB and basic CRUD operations"""
    try:
        # Connect to MongoDB
        print("Connecting to MongoDB...")
        client = MongoClient(MONGODB_URI)
        db = client["empowHER"]
        
        # Check connection
        print("Server info:", client.server_info())
        
        # Create test collections
        sessions_collection = db["test_sessions"]
        messages_collection = db["test_messages"]
        
        # Create a test session
        session_id = str(uuid.uuid4())
        print(f"Creating test session with ID: {session_id}")
        
        session_data = {
            "session_id": session_id,
            "user_id": "test_user",
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow(),
            "is_active": True
        }
        
        session_result = sessions_collection.insert_one(session_data)
        print(f"Inserted session with ID: {session_result.inserted_id}")
        
        # Create test messages
        message_data = {
            "role": "user",
            "content": "Hello, this is a test message",
            "session_id": session_id,
            "timestamp": datetime.datetime.utcnow()
        }
        
        message_result = messages_collection.insert_one(message_data)
        print(f"Inserted message with ID: {message_result.inserted_id}")
        
        # Query the data
        print("Finding session...")
        found_session = sessions_collection.find_one({"session_id": session_id})
        print("Found session:", found_session)
        
        print("Finding messages...")
        messages = list(messages_collection.find({"session_id": session_id}))
        print(f"Found {len(messages)} messages for session {session_id}")
        
        # Clean up
        print("Cleaning up test data...")
        sessions_collection.delete_one({"session_id": session_id})
        messages_collection.delete_many({"session_id": session_id})
        
        print("MongoDB connection test successful!")
        return True
        
    except Exception as e:
        print(f"Error testing MongoDB connection: {e}")
        return False

if __name__ == "__main__":
    test_mongodb_connection() 