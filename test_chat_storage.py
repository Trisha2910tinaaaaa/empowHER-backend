import os
from dotenv import load_dotenv
from pymongo import MongoClient
import datetime
import uuid
import time
import random

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

def test_chat_storage():
    """Test chat storage functionality with MongoDB"""
    try:
        # Connect to MongoDB
        print("Connecting to MongoDB...")
        client = MongoClient(MONGODB_URI)
        db = client["empowHER"]
        chat_sessions_collection = db["chat_sessions"]
        chat_messages_collection = db["chat_messages"]
        
        # Create test session
        user_id = f"test_user_{int(time.time())}"
        session_id = str(uuid.uuid4())
        print(f"Creating test session with ID: {session_id} for user: {user_id}")
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow(),
            "is_active": True
        }
        
        chat_sessions_collection.insert_one(session_data)
        
        # Simulate chat conversation
        user_messages = [
            "Hello, I'm looking for tech jobs.",
            "I have experience in Python and JavaScript.",
            "Are there any women-friendly companies hiring now?",
            "What skills are in demand for data science roles?",
            "Thank you for the information!"
        ]
        
        assistant_responses = [
            "Hi there! I'd be happy to help you find tech jobs. What kind of role are you looking for?",
            "Great skills! Python and JavaScript are very in-demand. Are you interested in frontend, backend, or full-stack positions?",
            "Yes, there are many women-friendly companies hiring now. Companies like Microsoft, Google, IBM, and Salesforce are known for their diversity initiatives. Would you like me to search for specific roles at these companies?",
            "For data science roles, the most in-demand skills include Python, SQL, machine learning frameworks like TensorFlow or PyTorch, data visualization tools like Tableau or PowerBI, and statistical analysis. Having cloud platform experience (AWS, Azure, GCP) is also valuable.",
            "You're welcome! Feel free to ask if you have any other questions about tech job opportunities. Good luck with your job search!"
        ]
        
        # Insert messages with timestamps
        print("Simulating chat conversation...")
        for i in range(5):
            # User message
            user_message_id = str(ObjectId())
            user_message = {
                "message_id": user_message_id,
                "session_id": session_id,
                "role": "user",
                "content": user_messages[i],
                "timestamp": datetime.datetime.utcnow()
            }
            chat_messages_collection.insert_one(user_message)
            print(f"User: {user_messages[i]}")
            
            # Wait a moment
            time.sleep(0.5)
            
            # Assistant response
            ai_message_id = str(ObjectId())
            ai_message = {
                "message_id": ai_message_id,
                "session_id": session_id,
                "role": "assistant",
                "content": assistant_responses[i],
                "timestamp": datetime.datetime.utcnow()
            }
            chat_messages_collection.insert_one(ai_message)
            print(f"Assistant: {assistant_responses[i]}")
            
            # Update session timestamp
            chat_sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": {
                    "updated_at": datetime.datetime.utcnow(),
                    "last_message_timestamp": datetime.datetime.utcnow()
                }}
            )
            
            # Wait between message pairs
            time.sleep(1)
        
        # Retrieve and verify conversation
        print("\nRetrieving conversation from database...")
        messages = list(chat_messages_collection.find(
            {"session_id": session_id},
            sort=[("timestamp", 1)]
        ))
        
        print(f"Retrieved {len(messages)} messages")
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg["timestamp"]
            print(f"{timestamp} - {role}: {content}")
        
        # Verify session info
        session = chat_sessions_collection.find_one({"session_id": session_id})
        print("\nSession information:")
        print(f"Session ID: {session['session_id']}")
        print(f"User ID: {session['user_id']}")
        print(f"Created at: {session['created_at']}")
        print(f"Last updated: {session['updated_at']}")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        return False

if __name__ == "__main__":
    from bson.objectid import ObjectId
    test_chat_storage() 