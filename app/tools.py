"""Mock support tools the agent can call. Replace the dicts with real systems."""
import json
import random
from typing import Any

ORDERS: dict[str, dict[str, Any]] = {
    "48213": {"status": "in transit", "carrier": "FastShip", "eta": "Thursday", "total": 59.98,
              "items": ["Wireless headphones", "USB-C cable"], "email": "customer@example.com"},
    "50977": {"status": "delivered", "carrier": "FastShip", "eta": "delivered Monday", "total": 29.99,
              "items": ["Premium plan, monthly"], "email": "customer@example.com"},
}

REFUND_LIMIT = 100.0

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "lookup_order", "description": "Look up an order by its number.",
        "parameters": {"type": "object", "properties": {
            "order_number": {"type": "string", "description": "Digits only, e.g. 48213"}},
            "required": ["order_number"]}}},
    {"type": "function", "function": {
        "name": "issue_refund",
        "description": "Refund an order. Refunds above 100 dollars must be escalated instead.",
        "parameters": {"type": "object", "properties": {
            "order_number": {"type": "string"},
            "amount": {"type": "number", "description": "Dollars"},
            "reason": {"type": "string"}},
            "required": ["order_number", "amount", "reason"]}}},
    {"type": "function", "function": {
        "name": "create_ticket",
        "description": "Open a follow-up ticket for the technical or billing team.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "normal", "high"]}},
            "required": ["summary"]}}},
]


def _lookup_order(order_number: str) -> dict[str, Any]:
    digits = "".join(c for c in str(order_number) if c.isdigit())
    order = ORDERS.get(digits)
    if order is None:
        return {"found": False, "order_number": digits}
    return {"found": True, "order_number": digits, **order}


def _issue_refund(order_number: str, amount: float, reason: str) -> dict[str, Any]:
    digits = "".join(c for c in str(order_number) if c.isdigit())
    order = ORDERS.get(digits)
    if order is None:
        return {"ok": False, "error": "order not found"}
    if amount <= 0 or amount > order["total"]:
        return {"ok": False, "error": f"amount must be between 0 and {order['total']}"}
    if amount > REFUND_LIMIT:
        return {"ok": False, "error": "above agent limit, create a high priority ticket"}
    return {"ok": True, "refund_id": f"RF-{random.randint(10000, 99999)}", "amount": amount,
            "eta_business_days": "5-7", "reason": reason}


def _create_ticket(summary: str, priority: str = "normal") -> dict[str, Any]:
    return {"ok": True, "ticket": f"TK{random.randint(10000, 99999)}", "priority": priority,
            "summary": summary}


_HANDLERS = {"lookup_order": _lookup_order, "issue_refund": _issue_refund,
             "create_ticket": _create_ticket}


def run_tool(name: str, raw_args: str) -> dict[str, Any]:
    """Execute a tool call; never raises, errors are returned to the model."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"ok": False, "error": f"unknown tool {name}"}
    try:
        args = json.loads(raw_args or "{}")
        return handler(**args)
    except (json.JSONDecodeError, TypeError) as exc:
        return {"ok": False, "error": f"bad arguments: {exc}"}
