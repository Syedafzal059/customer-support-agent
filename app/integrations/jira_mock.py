"""Mock Jira-style ticket lookup (Phase 6)."""

from __future__ import annotations


def get_ticket(ticket_id: str) -> dict[str, str]:
    """Return a fixed-shape mock payload for any normalized issue key."""
    tid = ticket_id.strip().upper()
    return {
        "id": tid,
        "status": "In Progress",
        "priority": "High",
        "summary": f"Login and access issue ({tid})",
    }
