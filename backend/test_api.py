"""
Simple test script to verify the API works
Run this after starting the server
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_submit_request():
    """Test submitting a request"""
    print("\n📝 Testing request submission...")
    
    payload = {
        "room_number": "101",
        "request_text": "I need two towels please",
        "intent": "towel_request"
    }
    
    response = requests.post(f"{BASE_URL}/api/submit-request", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return response.json()

def test_get_requests():
    """Test getting all requests"""
    print("\n📋 Testing get all requests...")
    
    response = requests.get(f"{BASE_URL}/api/requests")
    print(f"Status: {response.status_code}")
    print(f"Requests: {json.dumps(response.json(), indent=2)}")

def test_update_status(request_id):
    """Test updating request status"""
    print(f"\n✅ Testing status update for request {request_id}...")
    
    payload = {
        "request_id": request_id,
        "status": "completed"
    }
    
    response = requests.post(f"{BASE_URL}/api/update-status", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    print("🧪 Starting API Tests...")
    print(f"Server: {BASE_URL}")
    
    try:
        # Test 1: Submit request
        result = test_submit_request()
        request_id = result.get("request_id")
        
        # Test 2: Get all requests
        test_get_requests()
        
        # Test 3: Update status
        if request_id:
            test_update_status(request_id)
        
        # Test 4: Get requests again to see the update
        test_get_requests()
        
        print("\n✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to server.")
        print("Make sure the server is running: python -m app.main")
    except Exception as e:
        print(f"\n❌ Error: {e}")
