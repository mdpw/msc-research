import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "database.db"

def init_db():
    """Initialize database with requests table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

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
            completed_at TEXT
        )
    """)

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

    conn.commit()
    conn.close()
    print("Database initialized")

def add_request(room_number, request_text, department, intent=None):
    """Add a new request to database"""
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
