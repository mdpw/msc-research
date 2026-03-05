from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.models import RequestSubmit, RequestResponse, StatusUpdate, DepartmentUpdate, CancelRequest, StaffMessage, RatingSubmit
from app.database import (init_db, add_request, get_all_requests, update_request_status,
                           get_request_by_id, update_request_department, get_requests_by_room,
                           update_request_rating, add_staff_message,
                           get_all_departments, get_departments_detail, get_all_rooms,
                           get_intent_department_map, get_all_intent_mappings)
from pathlib import Path
import json

app = FastAPI(title="Hotel Voice Assistant API")

# Serve dashboard UI at /dashboard
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dashboard_connections = []
guest_connections = {}

@app.on_event("startup")
async def startup():
    init_db()
    intent_map = get_intent_department_map()
    departments = get_all_departments()
    print("Server started")
    print(f"Loaded {len(intent_map)} intent mappings from database")
    print(f"Available departments: {', '.join(departments)}")

@app.get("/")
async def root():
    intent_map = get_intent_department_map()
    departments = get_all_departments()
    return {
        "message": "Hotel Voice Assistant API",
        "status": "running",
        "intents_mapped": len(intent_map),
        "departments": departments
    }

@app.get("/api/departments")
async def get_departments_endpoint():
    """Get list of all departments"""
    return {"departments": get_all_departments()}

@app.get("/api/departments/detail")
async def get_departments_detail_endpoint():
    """Get all departments with descriptions"""
    return {"departments": get_departments_detail()}

@app.get("/api/rooms")
async def get_rooms():
    """Get list of all rooms"""
    return {"rooms": get_all_rooms()}

@app.get("/api/intent-mapping")
async def get_intent_mapping():
    """Get complete intent-to-department mapping (for debugging)"""
    return {"mappings": get_intent_department_map()}

@app.get("/api/intent-mapping/detail")
async def get_intent_mapping_detail():
    """Get all intent mappings with details"""
    return {"mappings": get_all_intent_mappings()}

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

    print(f"Request #{request_id} from Room {request.room_number}")
    print(f"   Text: {request.request_text}")
    print(f"   Intent: {request.intent}")
    print(f"   Department: {department}")

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

    print(f"Status updated: Request #{update.request_id} -> {update.status}")

    # Notify dashboards
    await notify_dashboards({
        "type": "status_update",
        "request_id": update.request_id,
        "status": update.status
    })

    # Notify guest device
    room_number = request_info["room_number"]

    status_messages = {
        "pending": "Your request has been received and is awaiting attention.",
        "in_progress": "Your request is being processed. We'll be with you shortly.",
        "completed": "Your request has been completed. Thank you for your patience!"
    }

    message = status_messages.get(update.status, "Your request status has been updated.")

    print(f"Notifying Room {room_number}: {message}")
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

    print(f"Request #{update.request_id}: {request_info['department']} -> {update.department}")

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

# ── Feature 5: Cancel Request ──
@app.post("/api/cancel-request")
async def cancel_request(cancel: CancelRequest):
    request_info = get_request_by_id(cancel.request_id)

    if not request_info:
        return {"success": False, "message": "Request not found"}

    if request_info["room_number"] != cancel.room_number:
        return {"success": False, "message": "Unauthorized"}

    if request_info["status"] == "completed":
        return {"success": False, "message": "Cannot cancel a completed request"}

    update_request_status(cancel.request_id, "cancelled")

    print(f"Request #{cancel.request_id} cancelled by Room {cancel.room_number}")

    await notify_dashboards({
        "type": "status_update",
        "request_id": cancel.request_id,
        "status": "cancelled"
    })

    return {"success": True, "message": "Request cancelled"}

# ── Feature 9: Staff Message to Guest ──
@app.post("/api/send-message")
async def send_message(msg: StaffMessage):
    request_info = get_request_by_id(msg.request_id)

    if not request_info:
        return {"success": False, "message": "Request not found"}

    message_id = add_staff_message(msg.request_id, msg.message, msg.staff_name)

    print(f"Staff message from {msg.staff_name} for Request #{msg.request_id}: {msg.message}")

    room_number = request_info["room_number"]
    await notify_guest(room_number, {
        "type": "staff_message",
        "request_id": msg.request_id,
        "message": msg.message,
        "staff_name": msg.staff_name
    })

    # Also notify dashboards so other staff see the message
    await notify_dashboards({
        "type": "staff_message",
        "request_id": msg.request_id,
        "message": msg.message,
        "staff_name": msg.staff_name
    })

    return {"success": True, "message": "Message sent", "message_id": message_id}

# ── Feature 10: Rate Request ──
@app.post("/api/rate-request")
async def rate_request(rating: RatingSubmit):
    request_info = get_request_by_id(rating.request_id)

    if not request_info:
        return {"success": False, "message": "Request not found"}

    if request_info["room_number"] != rating.room_number:
        return {"success": False, "message": "Unauthorized"}

    if rating.rating < 1 or rating.rating > 5:
        return {"success": False, "message": "Rating must be between 1 and 5"}

    update_request_rating(rating.request_id, rating.rating)

    print(f"Request #{rating.request_id} rated {rating.rating}/5 by Room {rating.room_number}")

    # Notify dashboards about the rating
    await notify_dashboards({
        "type": "rating_update",
        "request_id": rating.request_id,
        "rating": rating.rating
    })

    return {"success": True, "message": "Thank you for your feedback"}

@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    await websocket.accept()
    dashboard_connections.append(websocket)
    print(f"Dashboard connected. Total: {len(dashboard_connections)}")

    try:
        all_requests = get_all_requests()
        await websocket.send_json({"type": "initial", "requests": all_requests})

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        if websocket in dashboard_connections:
            dashboard_connections.remove(websocket)
        print(f"Dashboard disconnected. Remaining: {len(dashboard_connections)}")

@app.websocket("/ws/guest/{room_number}")
async def guest_websocket(websocket: WebSocket, room_number: str):
    await websocket.accept()
    guest_connections[room_number] = websocket
    print(f"Guest Room {room_number} connected. Total: {len(guest_connections)}")

    try:
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        if room_number in guest_connections and guest_connections[room_number] == websocket:
            del guest_connections[room_number]
        print(f"Guest Room {room_number} disconnected. Remaining: {len(guest_connections)}")

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
        print(f"Broadcasted '{data['type']}' to {len(dashboard_connections)} dashboard(s)")

async def notify_guest(room_number: str, data):
    """Notify specific guest room"""
    if room_number in guest_connections:
        try:
            await guest_connections[room_number].send_json(data)
            print(f"Notified Room {room_number}")
        except Exception as e:
            print(f"Failed to notify Room {room_number}: {e}")
            if room_number in guest_connections:
                del guest_connections[room_number]
    else:
        print(f"Room {room_number} not connected via WebSocket")

def route_to_department(text: str, intent: str = None) -> str:
    """
    Route requests to departments based on intent or text analysis
    Priority: intent (from DB mapping) > text keywords > default
    """

    # Priority 1: Use intent if available (most reliable) - lookup from DB
    if intent:
        intent_map = get_intent_department_map()
        if intent in intent_map:
            department = intent_map[intent]
            print(f"   Routed by intent: '{intent}' -> {department}")
            return department
        else:
            print(f"   Unknown intent: '{intent}', falling back to text analysis")

    # Priority 2: Fallback to text keyword analysis
    text_lower = text.lower()

    housekeeping_keywords = [
        "clean", "towel", "pillow", "blanket", "bed", "sheet", "laundry",
        "housekeeping", "tidy", "toiletries", "shampoo", "soap", "tissue",
        "toothbrush", "brush", "amenities"
    ]
    if any(word in text_lower for word in housekeeping_keywords):
        return "Housekeeping"

    room_service_keywords = [
        "food", "order", "breakfast", "lunch", "dinner", "menu",
        "water", "bottle", "coffee", "tea", "drink", "meal", "hungry",
        "snack", "beverage"
    ]
    if any(word in text_lower for word in room_service_keywords):
        return "Room Service"

    maintenance_keywords = [
        "temperature", "hot", "cold", "air conditioning", "ac", "heating",
        "broken", "not working", "light", "fix", "repair", "maintenance",
        "leak", "toilet", "shower", "tv", "remote"
    ]
    if any(word in text_lower for word in maintenance_keywords):
        return "Maintenance"

    concierge_keywords = [
        "taxi", "cab", "transport", "location", "direction", "recommend",
        "attraction", "restaurant", "tour", "booking"
    ]
    if any(word in text_lower for word in concierge_keywords):
        return "Concierge"

    front_desk_keywords = [
        "wake", "call", "checkout", "check out", "bill", "invoice",
        "noise", "complaint", "emergency", "help", "front desk"
    ]
    if any(word in text_lower for word in front_desk_keywords):
        return "Front Desk"

    print(f"   No match found, defaulting -> Front Desk")
    return "Front Desk"

@app.get("/dashboard")
async def serve_dashboard():
    """Serve the staff dashboard UI"""
    return FileResponse(FRONTEND_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
