from pydantic import BaseModel
from typing import Optional

class RequestSubmit(BaseModel):
    """Model for Android device submitting a request"""
    room_number: str
    request_text: str
    intent: Optional[str] = None
    
class RequestResponse(BaseModel):
    """Response after submitting request"""
    success: bool
    message: str
    request_id: int

class StatusUpdate(BaseModel):
    """Model for updating request status from dashboard"""
    request_id: int
    status: str  # 'pending', 'in_progress', 'completed'

class DepartmentUpdate(BaseModel):
    """Model for updating request department"""
    request_id: int
    department: str  # 'Housekeeping', 'Room Service', 'Maintenance', 'Front Desk', 'Concierge'

class Request(BaseModel):
    """Complete request model"""
    id: int
    room_number: str
    request_text: str
    intent: Optional[str] = None
    department: str
    status: str
    timestamp: str