"""
Lead capture tool for AutoStream.

In a production system this would POST to a CRM (HubSpot, Salesforce, etc.).
For this project it simulates a successful API call and prints the result.
"""

import re
from datetime import datetime


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Capture a qualified lead after collecting all required details.

    This function intentionally validates its inputs before "submitting"
    to reflect real-world API hygiene. It must NOT be called until all
    three fields have been collected from the user.

    Args:
        name:     Lead's full name.
        email:    Lead's email address.
        platform: Creator's primary platform (e.g. YouTube, Instagram).

    Returns:
        A dict with status, lead_id, and the submitted fields.

    Raises:
        ValueError: If any required field is missing or the email is malformed.
    """
    # --- Input validation ---
    if not name or not name.strip():
        raise ValueError("Lead name is required and cannot be empty.")
    if not email or not email.strip():
        raise ValueError("Lead email is required and cannot be empty.")
    if not platform or not platform.strip():
        raise ValueError("Creator platform is required and cannot be empty.")

    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    if not re.match(email_pattern, email.strip()):
        raise ValueError(f"Invalid email format: '{email}'")

    # --- Simulate CRM submission ---
    lead_id = f"LEAD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 52)
    print("  ✅  LEAD CAPTURED SUCCESSFULLY")
    print("=" * 52)
    print(f"  Lead ID  : {lead_id}")
    print(f"  Name     : {name.strip()}")
    print(f"  Email    : {email.strip()}")
    print(f"  Platform : {platform.strip()}")
    print(f"  Time     : {timestamp}")
    print("=" * 52 + "\n")

    return {
        "status": "success",
        "lead_id": lead_id,
        "name": name.strip(),
        "email": email.strip(),
        "platform": platform.strip(),
        "captured_at": timestamp,
    }
