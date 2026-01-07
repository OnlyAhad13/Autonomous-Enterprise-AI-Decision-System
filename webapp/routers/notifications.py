"""
Notifications Router - In-App Notification System.

Replaces Slack with real-time in-app notifications that the agent can send.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


# In-memory notification store (real-time)
notifications: List[Dict] = []
MAX_NOTIFICATIONS = 100


# ============================================================================
# Models
# ============================================================================

class Notification(BaseModel):
    """Single notification."""
    id: str
    type: str  # "info", "success", "warning", "error", "report"
    title: str
    message: str
    timestamp: str
    read: bool = False
    data: Optional[Dict] = None


class NotificationCreate(BaseModel):
    """Request to create a notification."""
    type: str = "info"
    title: str
    message: str
    data: Optional[Dict] = None


class ReportCreate(BaseModel):
    """Request to create a report notification."""
    title: str
    sections: List[Dict[str, str]]


# ============================================================================
# Endpoints
# ============================================================================

@router.get("")
async def get_notifications(unread_only: bool = False, limit: int = 50):
    """Get all notifications."""
    result = notifications
    
    if unread_only:
        result = [n for n in notifications if not n.get("read", False)]
    
    return {
        "notifications": result[:limit],
        "total": len(result),
        "unread_count": len([n for n in notifications if not n.get("read", False)]),
    }


@router.post("")
async def create_notification(notification: NotificationCreate):
    """Create a new notification."""
    notif = {
        "id": f"notif_{int(datetime.now().timestamp() * 1000)}",
        "type": notification.type,
        "title": notification.title,
        "message": notification.message,
        "timestamp": datetime.now().isoformat(),
        "read": False,
        "data": notification.data,
    }
    
    notifications.insert(0, notif)
    
    # Trim old notifications
    while len(notifications) > MAX_NOTIFICATIONS:
        notifications.pop()
    
    return notif


@router.post("/report")
async def create_report(report: ReportCreate):
    """Create a report notification (used by agent)."""
    # Format sections into message
    message_parts = []
    for section in report.sections:
        title = section.get("title", "")
        value = section.get("value", "")
        message_parts.append(f"**{title}:** {value}")
    
    message = "\n".join(message_parts)
    
    notif = {
        "id": f"report_{int(datetime.now().timestamp() * 1000)}",
        "type": "report",
        "title": report.title,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "read": False,
        "data": {"sections": report.sections},
    }
    
    notifications.insert(0, notif)
    
    while len(notifications) > MAX_NOTIFICATIONS:
        notifications.pop()
    
    return notif


@router.put("/{notification_id}/read")
async def mark_read(notification_id: str):
    """Mark a notification as read."""
    for notif in notifications:
        if notif["id"] == notification_id:
            notif["read"] = True
            return notif
    
    return {"error": "Notification not found"}


@router.put("/read-all")
async def mark_all_read():
    """Mark all notifications as read."""
    for notif in notifications:
        notif["read"] = True
    
    return {"message": "All notifications marked as read"}


@router.delete("/{notification_id}")
async def delete_notification(notification_id: str):
    """Delete a notification."""
    global notifications
    notifications = [n for n in notifications if n["id"] != notification_id]
    return {"message": "Notification deleted"}


@router.delete("")
async def clear_all():
    """Clear all notifications."""
    notifications.clear()
    return {"message": "All notifications cleared"}


# ============================================================================
# Helper function for agent to send notifications
# ============================================================================

def add_notification(
    type: str,
    title: str,
    message: str,
    data: Optional[Dict] = None,
) -> Dict:
    """Add a notification directly (used by agent tools)."""
    notif = {
        "id": f"notif_{int(datetime.now().timestamp() * 1000)}",
        "type": type,
        "title": title,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "read": False,
        "data": data,
    }
    
    notifications.insert(0, notif)
    
    while len(notifications) > MAX_NOTIFICATIONS:
        notifications.pop()
    
    return notif


def add_report(title: str, sections: List[Dict[str, str]]) -> Dict:
    """Add a report notification directly (used by agent tools)."""
    message_parts = []
    for section in sections:
        t = section.get("title", "")
        v = section.get("value", "")
        message_parts.append(f"**{t}:** {v}")
    
    return add_notification(
        type="report",
        title=title,
        message="\n".join(message_parts),
        data={"sections": sections},
    )
