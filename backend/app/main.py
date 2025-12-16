from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.models import RequestSubmit, RequestResponse, StatusUpdate, DepartmentUpdate
from app.database import init_db, add_request, get_all_requests, update_request_status, get_request_by_id, update_request_department
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

@app.on_event("startup")
async def startup():
    init_db()
    print("🚀 Server started")

@app.get("/")
async def root():
    return {"message": "Hotel Voice Assistant API", "status": "running"}

@app.post("/api/submit-request", response_model=RequestResponse)
async def submit_request(request: RequestSubmit):
    department = route_to_department(request.request_text, request.intent)
    
    request_id = add_request(
        room_number=request.room_number,
        request_text=request.request_text,
        department=department,
        intent=request.intent
    )
    
    print(f"✅ New request #{request_id} from Room {request.room_number}: {request.request_text}")
    
    # Get complete request data
    new_request = get_request_by_id(request_id)
    
    # FIXED: Only notify dashboards, don't send duplicate broadcasts
    if new_request:
        await notify_dashboards({
            "type": "new_request",
            **new_request
        })
    
    return RequestResponse(
        success=True,
        message=f"Your request has been sent to the {department}. We will serve you soon.",
        request_id=request_id
    )

@app.get("/api/requests")
async def get_requests():
    requests = get_all_requests()
    return {"requests": requests}

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
    
    if update.status == "in_progress":
        message = "Your request is acknowledged and we are working on it."
    elif update.status == "completed":
        message = "Your request is completed. Thank you and happy to serve you."
    else:
        message = "Your request status has been updated."
    
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
    update_request_department(update.request_id, update.department)
    
    print(f"🔄 Request #{update.request_id} moved to {update.department}")
    
    await notify_dashboards({
        "type": "department_update",
        "request_id": update.request_id,
        "department": update.department
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
        if room_number in guest_connections:
            del guest_connections[room_number]
        print(f"📱 Guest Room {room_number} disconnected. Remaining: {len(guest_connections)}")

async def notify_dashboards(data):
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
        print(f"📊 Broadcasted {data['type']} to {len(dashboard_connections)} dashboards")

async def notify_guest(room_number: str, data):
    if room_number in guest_connections:
        try:
            await guest_connections[room_number].send_json(data)
            print(f"✅ Notified Room {room_number}")
        except Exception as e:
            print(f"❌ Failed to notify Room {room_number}: {e}")
            if room_number in guest_connections:
                del guest_connections[room_number]
    else:
        print(f"⚠️ Room {room_number} not connected")

def route_to_department(text: str, intent: str = None) -> str:
    text_lower = text.lower()
    
    if intent:
        intent_to_dept = {
            "towel_request": "Housekeeping",
            "room_cleaning": "Housekeeping",
            "food_order": "Room Service",
            "temperature_control": "Maintenance",
            "maintenance_issue": "Maintenance",
            "maintenance": "Maintenance",
            "lighting_control": "Maintenance",
            "front_desk_query": "Front Desk",
            "misc_request": "Front Desk"
        }
        return intent_to_dept.get(intent, "Front Desk")
    
    if any(word in text_lower for word in ["towel", "clean", "bed", "pillow", "blanket"]):
        return "Housekeeping"
    elif any(word in text_lower for word in ["food", "order", "breakfast", "lunch", "dinner", "menu", "water", "bottle"]):
        return "Room Service"
    elif any(word in text_lower for word in ["temperature", "air conditioning", "heating", "broken", "not working", "light"]):
        return "Maintenance"
    else:
        return "Front Desk"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)