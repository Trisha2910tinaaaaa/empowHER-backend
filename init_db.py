import os
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

def init_mongodb():
    """Initialize MongoDB with required collections and indexes"""
    try:
        # Connect to MongoDB
        print("Connecting to MongoDB...")
        client = MongoClient(MONGODB_URI)
        db = client["empowHER"]
        
        # Create collections if they don't exist
        if "chat_sessions" not in db.list_collection_names():
            print("Creating chat_sessions collection...")
            db.create_collection("chat_sessions")
        
        if "chat_messages" not in db.list_collection_names():
            print("Creating chat_messages collection...")
            db.create_collection("chat_messages")
        
        # Set up indexes for chat_sessions
        print("Setting up indexes for chat_sessions...")
        sessions_collection = db["chat_sessions"]
        sessions_collection.create_index([("session_id", ASCENDING)], unique=True)
        sessions_collection.create_index([("user_id", ASCENDING)])
        sessions_collection.create_index([("updated_at", DESCENDING)])
        sessions_collection.create_index([("is_active", ASCENDING)])
        
        # Set up indexes for chat_messages
        print("Setting up indexes for chat_messages...")
        messages_collection = db["chat_messages"]
        messages_collection.create_index([("session_id", ASCENDING)])
        messages_collection.create_index([("message_id", ASCENDING)], unique=True)
        messages_collection.create_index([("timestamp", ASCENDING)])
        messages_collection.create_index([("role", ASCENDING)])
        messages_collection.create_index([("content", TEXT)])  # For text search
        
        print("MongoDB initialization completed successfully!")
        return True
    
    except Exception as e:
        print(f"Error initializing MongoDB: {e}")
        return False

if __name__ == "__main__":
    init_mongodb() 