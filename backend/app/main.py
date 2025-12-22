from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.models import RequestSubmit, RequestResponse, StatusUpdate, DepartmentUpdate
from app.database import init_db, add_request, get_all_requests, update_request_status, get_request_by_id, update_request_department, get_requests_by_room
import json

app = FastAPI(title="Hotel Voice Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dashboard_connections = []
guest_connections = {}

# COMPLETE INTENT-TO-DEPARTMENT MAPPING (All 18 intents from your NLU model)
INTENT_TO_DEPARTMENT = {
    # Housekeeping Department (7 intents)
    "room_cleaning": "Housekeeping",
    "towel_request": "Housekeeping",
    "toiletries_request": "Housekeeping",
    "blanket_request": "Housekeeping",
    "pillow_request": "Housekeeping",
    "laundry_service": "Housekeeping",
    "do_not_disturb": "Housekeeping",
    
    # Room Service / F&B (1 intent)
    "food_order": "Room Service",
    
    # Maintenance / Engineering (3 intents)
    "maintenance": "Maintenance",
    "temperature_control": "Maintenance",
    "lighting_control": "Maintenance",
    
    # Front Desk (4 intents)
    "wake_up_call": "Front Desk",
    "checkout_billing": "Front Desk",
    "noise_complaint": "Front Desk",
    "emergency": "Front Desk",
    
    # Concierge (2 intents)
    "concierge_general": "Concierge",
    "concierge_taxi": "Concierge",
    
    # Misc (1 intent)
    "misc_request": "Front Desk"
}

# Department display names (for consistency)
DEPARTMENTS = [
    "Housekeeping",
    "Room Service", 
    "Maintenance",
    "Front Desk",
    "Concierge"
]

@app.on_event("startup")
async def startup():
    init_db()
    print("🚀 Server started")
    print(f"📋 Loaded {len(INTENT_TO_DEPARTMENT)} intent mappings")
    print(f"🏨 Available departments: {', '.join(DEPARTMENTS)}")

@app.get("/")
async def root():
    return {
        "message": "Hotel Voice Assistant API", 
        "status": "running",
        "intents_mapped": len(INTENT_TO_DEPARTMENT),
        "departments": DEPARTMENTS
    }

@app.get("/api/departments")
async def get_departments():
    """Get list of all departments"""
    return {"departments": DEPARTMENTS}

@app.get("/api/intent-mapping")
async def get_intent_mapping():
    """Get complete intent-to-department mapping (for debugging)"""
    return {"mappings": INTENT_TO_DEPARTMENT}

@app.post("/api/submit-request", response_model=RequestResponse)
async def submit_request(request: RequestSubmit):
    # Route to department
    department = route_to_department(request.request_text, request.intent)
    
    request_id = add_request(
        room_number=request.room_number,
        request_text=request.request_text,
        department=department,
        intent=request.intent
    )
    
    print(f"✅ Request #{request_id} from Room {request.room_number}")
    print(f"   📝 Text: {request.request_text}")
    print(f"   🎯 Intent: {request.intent}")
    print(f"   🏢 Department: {department}")
    
    # Get complete request data
    new_request = get_request_by_id(request_id)
    
    # Notify dashboards
    if new_request:
        await notify_dashboards({
            "type": "new_request",
            **new_request
        })
    
    # Department-specific confirmation messages
    messages = {
        "Housekeeping": "Your housekeeping request has been received. Our team will assist you shortly.",
        "Room Service": "Your order has been received. We'll deliver it to your room soon.",
        "Maintenance": "Your maintenance request has been logged. A technician will address it shortly.",
        "Front Desk": "Your request has been received. The front desk will assist you shortly.",
        "Concierge": "Your request has been received. Our concierge will help you shortly.",
    }
    
    message = messages.get(department, f"Your request has been sent to {department}. We will serve you soon.")
    
    return RequestResponse(
        success=True,
        message=message,
        request_id=request_id
    )

@app.get("/api/requests")
async def get_requests():
    requests = get_all_requests()
    return {"requests": requests}

@app.get("/api/request-history")
async def get_request_history(room_number: str):
    """Get history of requests for a specific room"""
    requests = get_requests_by_room(room_number)
    return {"room_number": room_number, "requests": requests}

@app.post("/api/update-status")
async def update_status(update: StatusUpdate):
    request_info = get_request_by_id(update.request_id)
    
    if not request_info:
        return {"success": False, "message": "Request not found"}
    
    update_request_status(update.request_id, update.status)
    
    print(f"🔄 Status updated: Request #{update.request_id} → {update.status}")
    
    # Notify dashboards
    await notify_dashboards({
        "type": "status_update",
        "request_id": update.request_id,
        "status": update.status
    })
    
    # Notify guest device
    room_number = request_info["room_number"]
    
    # Status messages
    status_messages = {
        "pending": "Your request has been received and is awaiting attention.",
        "in_progress": "Your request is being processed. We'll be with you shortly.",
        "completed": "Your request has been completed. Thank you for your patience!"
    }
    
    message = status_messages.get(update.status, "Your request status has been updated.")
    
    print(f"📱 Notifying Room {room_number}: {message}")
    await notify_guest(room_number, {
        "type": "status_update",
        "request_id": update.request_id,
        "status": update.status,
        "message": message
    })
    
    return {"success": True, "message": "Status updated"}

@app.post("/api/update-department")
async def update_department(update: DepartmentUpdate):
    request_info = get_request_by_id(update.request_id)
    
    if not request_info:
        return {"success": False, "message": "Request not found"}
    
    update_request_department(update.request_id, update.department)
    
    print(f"🔄 Request #{update.request_id}: {request_info['department']} → {update.department}")
    
    await notify_dashboards({
        "type": "department_update",
        "request_id": update.request_id,
        "department": update.department
    })
    
    # Notify guest that department changed
    room_number = request_info["room_number"]
    await notify_guest(room_number, {
        "type": "department_update",
        "request_id": update.request_id,
        "message": f"Your request has been forwarded to {update.department}."
    })
    
    return {"success": True, "message": "Department updated"}

@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    await websocket.accept()
    dashboard_connections.append(websocket)
    print(f"📊 Dashboard connected. Total: {len(dashboard_connections)}")
    
    try:
        all_requests = get_all_requests()
        await websocket.send_json({"type": "initial", "requests": all_requests})
        
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        if websocket in dashboard_connections:
            dashboard_connections.remove(websocket)
        print(f"📊 Dashboard disconnected. Remaining: {len(dashboard_connections)}")

@app.websocket("/ws/guest/{room_number}")
async def guest_websocket(websocket: WebSocket, room_number: str):
    await websocket.accept()
    guest_connections[room_number] = websocket
    print(f"📱 Guest Room {room_number} connected. Total: {len(guest_connections)}")
    
    try:
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        if room_number in guest_connections and guest_connections[room_number] == websocket:
            del guest_connections[room_number]
        print(f"📱 Guest Room {room_number} disconnected. Remaining: {len(guest_connections)}")

async def notify_dashboards(data):
    """Notify all connected dashboards"""
    disconnected = []
    for connection in dashboard_connections:
        try:
            await connection.send_json(data)
        except:
            disconnected.append(connection)
    
    for conn in disconnected:
        if conn in dashboard_connections:
            dashboard_connections.remove(conn)
    
    if data.get("type"):
        print(f"📊 Broadcasted '{data['type']}' to {len(dashboard_connections)} dashboard(s)")

async def notify_guest(room_number: str, data):
    """Notify specific guest room"""
    if room_number in guest_connections:
        try:
            await guest_connections[room_number].send_json(data)
            print(f"✅ Notified Room {room_number}")
        except Exception as e:
            print(f"❌ Failed to notify Room {room_number}: {e}")
            if room_number in guest_connections:
                del guest_connections[room_number]
    else:
        print(f"⚠️ Room {room_number} not connected via WebSocket")

def route_to_department(text: str, intent: str = None) -> str:
    """
    Route requests to departments based on intent or text analysis
    Priority: intent > text keywords > default
    """
    
    # Priority 1: Use intent if available (most reliable)
    if intent:
        if intent in INTENT_TO_DEPARTMENT:
            department = INTENT_TO_DEPARTMENT[intent]
            print(f"   ✓ Routed by intent: '{intent}' → {department}")
            return department
        else:
            print(f"   ⚠️ Unknown intent: '{intent}', falling back to text analysis")
    
    # Priority 2: Fallback to text keyword analysis
    text_lower = text.lower()
    
    # Housekeeping keywords
    housekeeping_keywords = [
        "clean", "towel", "pillow", "blanket", "bed", "sheet", "laundry",
        "housekeeping", "tidy", "toiletries", "shampoo", "soap", "tissue",
        "toothbrush", "brush", "amenities"
    ]
    if any(word in text_lower for word in housekeeping_keywords):
        print(f"   ✓ Routed by keywords → Housekeeping")
        return "Housekeeping"
    
    # Room Service keywords
    room_service_keywords = [
        "food", "order", "breakfast", "lunch", "dinner", "menu", 
        "water", "bottle", "coffee", "tea", "drink", "meal", "hungry",
        "snack", "beverage"
    ]
    if any(word in text_lower for word in room_service_keywords):
        print(f"   ✓ Routed by keywords → Room Service")
        return "Room Service"
    
    # Maintenance keywords
    maintenance_keywords = [
        "temperature", "hot", "cold", "air conditioning", "ac", "heating",
        "broken", "not working", "light", "fix", "repair", "maintenance",
        "leak", "toilet", "shower", "tv", "remote"
    ]
    if any(word in text_lower for word in maintenance_keywords):
        print(f"   ✓ Routed by keywords → Maintenance")
        return "Maintenance"
    
    # Concierge keywords
    concierge_keywords = [
        "taxi", "cab", "transport", "location", "direction", "recommend",
        "attraction", "restaurant", "tour", "booking"
    ]
    if any(word in text_lower for word in concierge_keywords):
        print(f"   ✓ Routed by keywords → Concierge")
        return "Concierge"
    
    # Front Desk keywords
    front_desk_keywords = [
        "wake", "call", "checkout", "check out", "bill", "invoice",
        "noise", "complaint", "emergency", "help", "front desk"
    ]
    if any(word in text_lower for word in front_desk_keywords):
        print(f"   ✓ Routed by keywords → Front Desk")
        return "Front Desk"
    
    # Default: Front Desk
    print(f"   ⚠️ No match found, defaulting → Front Desk")
    return "Front Desk"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)