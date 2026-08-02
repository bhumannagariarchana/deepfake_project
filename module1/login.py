import uuid
import datetime

# Mock database of valid credentials
VALID_CREDENTIALS = {
    "admin": "admin123",
    "user123": "password123",
    "employee_007": "bond007"
}

VALID_EMPLOYEE_IDS = {"EMP101", "EMP102", "EMP103", "EMP104", "EMP105"}

def validate_login(username_or_id, password=None):
    """
    Validates a login request either by username/password or by Employee ID alone.
    Returns (success: bool, user_identifier: str, error_message: str)
    """
    if not username_or_id:
        return False, "", "Identifier (Username or Employee ID) cannot be empty."

    # Case 1: Employee ID login (if password is not provided or is empty)
    if not password:
        if username_or_id in VALID_EMPLOYEE_IDS:
            return True, username_or_id, ""
        else:
            return False, "", f"Invalid Employee ID: {username_or_id}"

    # Case 2: Username and Password login
    if username_or_id in VALID_CREDENTIALS:
        if VALID_CREDENTIALS[username_or_id] == password:
            return True, username_or_id, ""
        else:
            return False, "", "Incorrect password."
    else:
        # Check if the identifier is a valid Employee ID even if a password was provided
        if username_or_id in VALID_EMPLOYEE_IDS:
            return True, username_or_id, ""
        return False, "", f"Username '{username_or_id}' not found."

def create_session(user_identifier):
    """
    Initializes a new liveness session metadata structure.
    """
    session_id = str(uuid.uuid4())
    return {
        "session_id": session_id,
        "user_id": user_identifier,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "initialized"
    }
