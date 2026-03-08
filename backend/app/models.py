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
    department: str

class StatusUpdate(BaseModel):
    """Model for updating request status from dashboard"""
    request_id: int
    status: str  # 'pending', 'in_progress', 'completed'

class DepartmentUpdate(BaseModel):
    """Model for updating request department"""
    request_id: int
    department: str  # 'Housekeeping', 'Room Service', 'Maintenance', 'Front Desk', 'Concierge'

class CancelRequest(BaseModel):
    """Model for cancelling a request from guest device"""
    request_id: int
    room_number: str

class StaffMessage(BaseModel):
    """Model for staff sending a message to guest"""
    request_id: int
    message: str
    staff_name: str

class RatingSubmit(BaseModel):
    """Model for guest rating a completed request"""
    request_id: int
    rating: int  # 1-5
    room_number: str

class Request(BaseModel):
    """Complete request model"""
    id: int
    room_number: str
    request_text: str
    intent: Optional[str] = None
    department: str
    status: str
    timestamp: str
