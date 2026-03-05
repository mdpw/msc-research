import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "database.db"

def get_connection():
    """Get a database connection with foreign keys enabled"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    """Initialize database with all tables"""
    conn = get_connection()
    cursor = conn.cursor()

    # ── Rooms table ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_number TEXT NOT NULL UNIQUE,
            floor INTEGER NOT NULL,
            room_type TEXT NOT NULL DEFAULT 'Standard'
        )
    """)

    # ── Departments table ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT
        )
    """)

    # ── Intent-to-Department mapping table ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS intent_department_mapping (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intent TEXT NOT NULL UNIQUE,
            department_name TEXT NOT NULL,
            FOREIGN KEY (department_name) REFERENCES departments(name)
        )
    """)

    # ── Requests table (with foreign keys to rooms and departments) ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_number TEXT NOT NULL,
            request_text TEXT NOT NULL,
            intent TEXT,
            department TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            rating INTEGER,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            FOREIGN KEY (room_number) REFERENCES rooms(room_number),
            FOREIGN KEY (department) REFERENCES departments(name)
        )
    """)

    # ── Staff messages table ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS staff_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            staff_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (request_id) REFERENCES requests(id)
        )
    """)

    # Add rating column if it doesn't exist (migration for existing DBs)
    try:
        cursor.execute("ALTER TABLE requests ADD COLUMN rating INTEGER")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # ── Seed departments ──
    departments = [
        ("Housekeeping", "Handles room cleaning, towels, toiletries, laundry, and bedding requests"),
        ("Room Service", "Handles food and beverage orders delivered to guest rooms"),
        ("Maintenance", "Handles repairs, temperature control, lighting, and technical issues"),
        ("Front Desk", "Handles check-in/out, billing, wake-up calls, complaints, and emergencies"),
        ("Concierge", "Handles taxi bookings, local recommendations, and general guest inquiries"),
    ]
    for name, description in departments:
        cursor.execute(
            "INSERT OR IGNORE INTO departments (name, description) VALUES (?, ?)",
            (name, description)
        )

    # ── Seed intent-to-department mappings ──
    intent_mappings = [
        # Housekeeping (7 intents)
        ("room_cleaning", "Housekeeping"),
        ("towel_request", "Housekeeping"),
        ("toiletries_request", "Housekeeping"),
        ("blanket_request", "Housekeeping"),
        ("pillow_request", "Housekeeping"),
        ("laundry_service", "Housekeeping"),
        ("do_not_disturb", "Housekeeping"),
        # Room Service (1 intent)
        ("food_order", "Room Service"),
        # Maintenance (3 intents)
        ("maintenance", "Maintenance"),
        ("temperature_control", "Maintenance"),
        ("lighting_control", "Maintenance"),
        # Front Desk (4 intents)
        ("wake_up_call", "Front Desk"),
        ("checkout_billing", "Front Desk"),
        ("noise_complaint", "Front Desk"),
        ("emergency", "Front Desk"),
        # Concierge (2 intents)
        ("concierge_general", "Concierge"),
        ("concierge_taxi", "Concierge"),
        # Misc (1 intent)
        ("misc_request", "Front Desk"),
    ]
    for intent, dept in intent_mappings:
        cursor.execute(
            "INSERT OR IGNORE INTO intent_department_mapping (intent, department_name) VALUES (?, ?)",
            (intent, dept)
        )

    # ── Seed rooms (floors 1-3, rooms 101-310) ──
    rooms = [
        # Floor 1
        ("101", 1, "Standard"), ("102", 1, "Standard"), ("103", 1, "Standard"),
        ("104", 1, "Deluxe"), ("105", 1, "Deluxe"),
        # Floor 2
        ("201", 2, "Standard"), ("202", 2, "Standard"), ("203", 2, "Standard"),
        ("204", 2, "Deluxe"), ("205", 2, "Deluxe"),
        # Floor 3
        ("301", 3, "Suite"), ("302", 3, "Suite"),
        ("303", 3, "Deluxe"), ("304", 3, "Deluxe"), ("305", 3, "Standard"),
    ]
    for room_number, floor, room_type in rooms:
        cursor.execute(
            "INSERT OR IGNORE INTO rooms (room_number, floor, room_type) VALUES (?, ?, ?)",
            (room_number, floor, room_type)
        )

    conn.commit()
    conn.close()
    print("Database initialized")

# ── Room functions ──

def get_all_rooms():
    """Get all rooms"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, room_number, floor, room_type FROM rooms ORDER BY room_number")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "room_number": r[1], "floor": r[2], "room_type": r[3]} for r in rows]

def get_room(room_number):
    """Get a specific room by room_number"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, room_number, floor, room_type FROM rooms WHERE room_number = ?", (room_number,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "room_number": row[1], "floor": row[2], "room_type": row[3]}
    return None

# ── Department functions ──

def get_all_departments():
    """Get all department names"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM departments ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]

def get_departments_detail():
    """Get all departments with details"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, description FROM departments ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "description": r[2]} for r in rows]

# ── Intent mapping functions ──

def get_intent_department_map():
    """Get intent-to-department mapping as a dictionary"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT intent, department_name FROM intent_department_mapping")
    rows = cursor.fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}

def get_all_intent_mappings():
    """Get all intent mappings with details"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, intent, department_name FROM intent_department_mapping ORDER BY department_name, intent")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "intent": r[1], "department_name": r[2]} for r in rows]

# ── Request functions ──

def add_request(room_number, request_text, department, intent=None):
    """Add a new request to database"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO requests (room_number, request_text, intent, department, status, created_at)
        VALUES (?, ?, ?, ?, 'pending', ?)
    """, (room_number, request_text, intent, department, datetime.now().isoformat()))

    request_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return request_id

def get_all_requests():
    """Get all requests ordered by newest first"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, room_number, request_text, intent, department, status, created_at, rating
        FROM requests
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    requests = []
    for row in rows:
        requests.append({
            "id": row[0],
            "room_number": row[1],
            "request_text": row[2],
            "intent": row[3],
            "department": row[4],
            "status": row[5],
            "created_at": row[6],
            "rating": row[7]
        })

    return requests

def get_request_by_id(request_id):
    """Get a specific request by ID"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, room_number, request_text, intent, department, status, created_at, rating
        FROM requests
        WHERE id = ?
    """, (request_id,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "id": row[0],
            "room_number": row[1],
            "request_text": row[2],
            "intent": row[3],
            "department": row[4],
            "status": row[5],
            "created_at": row[6],
            "rating": row[7]
        }
    return None

def update_request_status(request_id, status):
    """Update request status"""
    conn = get_connection()
    cursor = conn.cursor()

    completed_at = datetime.now().isoformat() if status == 'completed' else None

    cursor.execute("""
        UPDATE requests
        SET status = ?, completed_at = ?
        WHERE id = ?
    """, (status, completed_at, request_id))

    conn.commit()
    conn.close()

def update_request_department(request_id, department):
    """Update request department"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE requests
        SET department = ?
        WHERE id = ?
    """, (department, request_id))

    conn.commit()
    conn.close()

def update_request_rating(request_id, rating):
    """Update request rating (1-5)"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE requests
        SET rating = ?
        WHERE id = ?
    """, (rating, request_id))

    conn.commit()
    conn.close()

def add_staff_message(request_id, message, staff_name):
    """Add a staff message for a request"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO staff_messages (request_id, message, staff_name, created_at)
        VALUES (?, ?, ?, ?)
    """, (request_id, message, staff_name, datetime.now().isoformat()))

    message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return message_id

def get_messages_for_request(request_id):
    """Get all staff messages for a request"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, request_id, message, staff_name, created_at
        FROM staff_messages
        WHERE request_id = ?
        ORDER BY created_at ASC
    """, (request_id,))

    rows = cursor.fetchall()
    conn.close()

    return [{"id": r[0], "request_id": r[1], "message": r[2], "staff_name": r[3], "created_at": r[4]} for r in rows]

def get_requests_by_room(room_number):
    """Get all requests for a specific room"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, room_number, request_text, intent, department, status, created_at, rating
        FROM requests
        WHERE room_number = ?
        ORDER BY created_at DESC
    """, (room_number,))

    rows = cursor.fetchall()
    conn.close()

    requests = []
    for row in rows:
        requests.append({
            "id": row[0],
            "room_number": row[1],
            "request_text": row[2],
            "intent": row[3],
            "department": row[4],
            "status": row[5],
            "created_at": row[6],
            "rating": row[7]
        })

    return requests
