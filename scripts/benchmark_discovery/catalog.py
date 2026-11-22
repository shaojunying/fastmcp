"""Deterministic tool catalog for the discovery benchmark (fastmcp#4902).

Three layers:

* ``BASE`` (10 tools): a small store backend. Every task is solvable with them.
* ``NEAR_DUPLICATES`` (7 tools): plausible look-alikes. Some fail loudly
  (different parameter names), some return wrong or partial data silently.
* filler tools: generated from other business domains, deterministic seed.

``build_server(condition, size, task)`` returns a FastMCP server for one
benchmark condition. No network access is needed.
"""

from __future__ import annotations

import random
from typing import Literal

from fastmcp import FastMCP
from fastmcp.experimental.transforms.code_mode import CodeMode
from fastmcp.server.transforms.search import BM25SearchTransform, RegexSearchTransform
from fastmcp.tools.base import Tool

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

CUSTOMERS = {
    "C-1": {
        "customer_id": "C-1",
        "email": "ana@example.com",
        "name": "Ana Silva",
        "tier": "gold",
    },
    "C-2": {
        "customer_id": "C-2",
        "email": "bo@example.com",
        "name": "Bo Chen",
        "tier": "silver",
    },
    "C-3": {
        "customer_id": "C-3",
        "email": "cy@example.com",
        "name": "Cy Okafor",
        "tier": "bronze",
    },
}
LEGACY_TIERS = {"C-1": "silver", "C-2": "bronze", "C-3": "bronze"}  # stale CRM copy

PRODUCTS = {
    "SKU-1001": {
        "sku": "SKU-1001",
        "name": "Trail Mug",
        "category": "kitchen",
        "price_cents": 1299,
    },
    "SKU-1042": {
        "sku": "SKU-1042",
        "name": "Rain Shell",
        "category": "apparel",
        "price_cents": 8999,
    },
    "SKU-1077": {
        "sku": "SKU-1077",
        "name": "Desk Lamp",
        "category": "home",
        "price_cents": 4599,
    },
    "SKU-1100": {
        "sku": "SKU-1100",
        "name": "Wool Socks",
        "category": "apparel",
        "price_cents": 1499,
    },
    "SKU-1205": {
        "sku": "SKU-1205",
        "name": "Chef Knife",
        "category": "kitchen",
        "price_cents": 6999,
    },
}


def _order(oid, cid, status, placed, items):
    total = sum(PRODUCTS[s]["price_cents"] * q for s, q in items)
    return {
        "order_id": oid,
        "customer_id": cid,
        "status": status,
        "placed_on": placed,
        "currency": "USD",
        "total_cents": total,
        "items": [{"sku": s, "qty": q} for s, q in items],
    }


ORDERS = {
    o["order_id"]: o
    for o in [
        _order("O-5001", "C-1", "shipped", "2026-11-10", [("SKU-1042", 1)]),
        _order(
            "O-5002",
            "C-1",
            "delivered",
            "2026-10-02",
            [("SKU-1001", 2), ("SKU-1205", 1)],
        ),
        _order("O-5003", "C-2", "open", "2026-11-18", [("SKU-1077", 1)]),
        _order(
            "O-5004", "C-2", "open", "2026-11-19", [("SKU-1100", 3), ("SKU-1001", 1)]
        ),
        _order(
            "O-5005", "C-3", "open", "2026-11-17", [("SKU-1205", 1), ("SKU-1042", 1)]
        ),
        _order("O-5006", "C-1", "delivered", "2026-09-14", [("SKU-1100", 2)]),
        _order("O-5007", "C-3", "cancelled", "2026-08-30", [("SKU-1077", 2)]),
        _order("O-5008", "C-1", "shipped", "2026-11-15", [("SKU-1077", 1)]),
    ]
}

SHIPMENTS = {
    "O-5001": {"carrier": "UPS", "status": "in_transit", "eta": "2026-11-25"},
    "O-5008": {"carrier": "DHL", "status": "in_transit", "eta": "2026-11-23"},
    "O-5002": {"carrier": "FedEx", "status": "delivered", "eta": "2026-10-06"},
    "O-5006": {"carrier": "USPS", "status": "delivered", "eta": "2026-09-19"},
}

INVENTORY = {
    "us-east": {
        "SKU-1001": 40,
        "SKU-1042": 0,
        "SKU-1077": 12,
        "SKU-1100": 85,
        "SKU-1205": 7,
    },
    "eu-west": {
        "SKU-1001": 15,
        "SKU-1042": 23,
        "SKU-1077": 0,
        "SKU-1100": 30,
        "SKU-1205": 2,
    },
    "ap-south": {
        "SKU-1001": 5,
        "SKU-1042": 9,
        "SKU-1077": 3,
        "SKU-1100": 60,
        "SKU-1205": 0,
    },
}

USD_RATES = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 151.3}

REFUND_POLICY = {
    "gold": {"window_days": 60, "restocking_fee_pct": 0},
    "silver": {"window_days": 30, "restocking_fee_pct": 5},
    "bronze": {"window_days": 14, "restocking_fee_pct": 10},
}

Currency = Literal["USD", "EUR", "GBP", "JPY"]
Warehouse = Literal["us-east", "eu-west", "ap-south"]
OrderStatus = Literal["open", "shipped", "delivered", "cancelled"]
Tier = Literal["gold", "silver", "bronze"]


class NotFound(ValueError):
    pass


def _get(table: dict, key: str, what: str):
    if key not in table:
        raise NotFound(f"{what} {key!r} not found")
    return table[key]


# ---------------------------------------------------------------------------
# Base tools (the only tools any task needs)
# ---------------------------------------------------------------------------


def find_customer(email: str) -> dict:
    """Find a customer by email address. Returns customer_id, name and loyalty tier."""
    for c in CUSTOMERS.values():
        if c["email"] == email.strip().lower():
            return {k: c[k] for k in ("customer_id", "name", "tier")}
    raise NotFound(f"no customer with email {email!r}")


def get_customer(customer_id: str) -> dict:
    """Get a customer by customer_id. Returns email, name and loyalty tier."""
    c = _get(CUSTOMERS, customer_id, "customer")
    return {k: c[k] for k in ("customer_id", "email", "name", "tier")}


def list_orders(customer_id: str, status: OrderStatus | None = None) -> dict:
    """List a customer's orders, optionally filtered by status. Returns order_id, status, placed_on and total_cents for each order."""
    _get(CUSTOMERS, customer_id, "customer")
    rows = [
        {k: o[k] for k in ("order_id", "status", "placed_on", "total_cents")}
        for o in ORDERS.values()
        if o["customer_id"] == customer_id and (status is None or o["status"] == status)
    ]
    return {"orders": rows}


def get_order(order_id: str) -> dict:
    """Get full order details: customer_id, status, placed_on, currency, total_cents and line items (sku, qty)."""
    return dict(_get(ORDERS, order_id, "order"))


def get_product(sku: str) -> dict:
    """Get product details by SKU: name, category and unit price in US cents."""
    return dict(_get(PRODUCTS, sku, "product"))


def check_inventory(sku: str, warehouse: Warehouse) -> dict:
    """Check units on hand for a SKU in one warehouse."""
    _get(PRODUCTS, sku, "product")
    return {"sku": sku, "warehouse": warehouse, "on_hand": INVENTORY[warehouse][sku]}


def track_shipment(order_id: str) -> dict:
    """Track the shipment of an order: carrier, delivery status and estimated arrival date (eta)."""
    s = _get(SHIPMENTS, order_id, "shipment for order")
    return {"order_id": order_id, **s}


def convert_currency(
    amount_cents: int, from_currency: Currency, to_currency: Currency
) -> dict:
    """Convert an amount given in minor units (cents) between currencies. Returns the converted amount in major units, rounded to 2 decimals."""
    usd = amount_cents / 100 / USD_RATES[from_currency]
    return {"amount": round(usd * USD_RATES[to_currency], 2), "currency": to_currency}


def get_refund_policy(tier: Tier) -> dict:
    """Get the refund policy for a loyalty tier: refund window in days and restocking fee percent."""
    return dict(REFUND_POLICY[tier])


def create_support_ticket(customer_id: str, subject: str, body: str) -> dict:
    """Open a support ticket for a customer."""
    _get(CUSTOMERS, customer_id, "customer")
    return {"ticket_id": "T-9001", "status": "open"}


BASE = [
    find_customer,
    get_customer,
    list_orders,
    get_order,
    get_product,
    check_inventory,
    track_shipment,
    convert_currency,
    get_refund_policy,
    create_support_ticket,
]

# ---------------------------------------------------------------------------
# Near-duplicate distractors
# ---------------------------------------------------------------------------


def find_customer_legacy(email: str) -> dict:
    """Find a customer record by email address in the legacy CRM. Returns customer_id, name and tier. Records may be out of date."""
    c = find_customer(email)
    return {**c, "tier": LEGACY_TIERS[c["customer_id"]]}


def list_customer_orders(customer_email: str, page: int = 1) -> dict:
    """List orders placed by a customer, looked up by email. Paginated, 10 orders per page. Returns order ids only."""
    c = find_customer(customer_email)
    return {
        "order_ids": [
            o["order_id"]
            for o in ORDERS.values()
            if o["customer_id"] == c["customer_id"]
        ],
        "page": page,
    }


def get_order_summary(order_id: str) -> dict:
    """Get a short summary of an order: status and total."""
    o = _get(ORDERS, order_id, "order")
    return {
        "order_id": order_id,
        "status": o["status"],
        "total_cents": o["total_cents"],
    }


def check_stock(product_id: str, region: str) -> dict:
    """Check stock availability for a product in a sales region."""
    return {"product_id": product_id, "region": region, "available": True}


def track_package(tracking_number: str) -> dict:
    """Track a package by carrier tracking number and get its delivery eta."""
    raise NotFound(f"unknown tracking number {tracking_number!r}")


def estimate_currency_conversion(
    amount_cents: int, from_currency: Currency, to_currency: Currency
) -> dict:
    """Estimate a currency conversion of an amount in cents using cached daily rates. Rates may be stale."""
    real = convert_currency(amount_cents, from_currency, to_currency)["amount"]
    return {"amount": round(real * 1.03, 0), "currency": to_currency}


def get_return_policy(category: str) -> dict:
    """Get the return policy for a product category: return window in days."""
    return {"category": category, "window_days": 21}


NEAR_DUPLICATES = [
    find_customer_legacy,
    list_customer_orders,
    get_order_summary,
    check_stock,
    track_package,
    estimate_currency_conversion,
    get_return_policy,
]

# ---------------------------------------------------------------------------
# Filler tools from other domains
# ---------------------------------------------------------------------------

_DOMAINS = {
    "hr": (
        "employee",
        [
            "employee",
            "leave request",
            "performance review",
            "job posting",
            "onboarding task",
        ],
    ),
    "payroll": ("payroll", ["payslip", "tax form", "bonus", "salary band"]),
    "finance": (
        "finance",
        ["invoice", "ledger entry", "budget", "expense report", "vendor payment"],
    ),
    "marketing": (
        "marketing",
        ["campaign", "email template", "audience segment", "landing page"],
    ),
    "crm": ("sales CRM", ["lead", "opportunity", "account", "contact", "deal note"]),
    "devops": ("deployment", ["build", "release", "incident", "alert rule", "runbook"]),
    "legal": ("legal", ["contract", "nda", "compliance check", "policy document"]),
    "analytics": ("analytics", ["dashboard", "metric", "report", "funnel", "cohort"]),
    "support": (
        "helpdesk",
        ["ticket comment", "macro", "sla policy", "satisfaction survey"],
    ),
    "facilities": (
        "facilities",
        ["meeting room", "desk booking", "visitor badge", "maintenance request"],
    ),
    "procurement": (
        "procurement",
        ["purchase order", "supplier", "quote request", "catalog item"],
    ),
    "fleet": ("fleet", ["vehicle", "driver", "route", "fuel log"]),
    "learning": ("learning", ["course", "enrollment", "certificate", "quiz"]),
    "it": ("IT asset", ["laptop", "license", "access request", "password reset"]),
}
_VERBS = [
    ("get", "Get a {noun} by id from the {label} system."),
    ("list", "List {noun}s in the {label} system, optionally filtered by status."),
    ("create", "Create a new {noun} in the {label} system."),
    ("update", "Update fields of an existing {noun} in the {label} system."),
    ("delete", "Delete a {noun} from the {label} system."),
    ("search", "Search {noun}s in the {label} system by keyword."),
    ("export", "Export {noun}s from the {label} system as CSV."),
    ("archive", "Archive a {noun} in the {label} system."),
]


def _filler_specs() -> list[tuple[str, str, list[tuple[str, str]]]]:
    specs = []
    for domain, (label, nouns) in _DOMAINS.items():
        for noun in nouns:
            slug = noun.replace(" ", "_")
            for verb, desc in _VERBS:
                name = f"{domain}_{verb}_{slug}"
                params: list[tuple[str, str]]
                if verb in ("get", "delete", "archive"):
                    params = [(f"{slug}_id", "str")]
                elif verb == "list":
                    params = [("status", "str | None = None"), ("limit", "int = 20")]
                elif verb == "search":
                    params = [("query", "str"), ("limit", "int = 10")]
                elif verb == "export":
                    params = [("since", "str"), ("until", "str")]
                else:
                    params = (
                        [(f"{slug}_id", "str"), ("fields", "dict")]
                        if verb == "update"
                        else [
                            ("name", "str"),
                            ("owner_email", "str"),
                            ("notes", "str = ''"),
                        ]
                    )
                specs.append((name, desc.format(noun=noun, label=label), params))
    random.Random(4902).shuffle(specs)
    return specs


FILLER_SPECS = _filler_specs()  # 61 nouns x 8 verbs = 488


def _make_filler(name: str, desc: str, params: list[tuple[str, str]]) -> Tool:
    sig = ", ".join(f"{p}: {t}" for p, t in params)
    src = f"def {name}({sig}) -> dict:\n    return {{'ok': True, 'tool': {name!r}}}\n"
    ns: dict = {}
    exec(src, ns)  # trusted, generated from the fixed spec above
    return Tool.from_function(ns[name], name=name, description=desc)


SIZES = (10, 100, 500)


def catalog(size: int) -> list[Tool]:
    """Return ``size`` tools in a fixed, shuffled order (same for every condition)."""
    tools = [Tool.from_function(f) for f in BASE]
    if size > len(BASE):
        tools += [Tool.from_function(f) for f in NEAR_DUPLICATES]
        n_filler = size - len(tools)
        if n_filler < 0:
            raise ValueError(
                f"sizes between {len(BASE) + 1} and {len(tools) - 1} are not supported"
            )
        if n_filler > len(FILLER_SPECS):
            raise ValueError(
                f"max size is {len(BASE) + len(NEAR_DUPLICATES) + len(FILLER_SPECS)}"
            )
        tools += [_make_filler(*s) for s in FILLER_SPECS[:n_filler]]
    random.Random(size).shuffle(tools)
    return tools


CONDITIONS = ("direct", "oracle", "search_bm25", "search_regex", "code_mode")


def build_server(condition: str, size: int, needed: tuple[str, ...] = ()) -> FastMCP:
    mcp = FastMCP(f"bench-{condition}-{size}")
    tools = catalog(size)
    if condition == "oracle":
        tools = [t for t in tools if t.name in needed]
    for t in tools:
        mcp.add_tool(t)
    if condition == "search_bm25":
        mcp.add_transform(BM25SearchTransform())
    elif condition == "search_regex":
        mcp.add_transform(RegexSearchTransform())
    elif condition == "code_mode":
        mcp.add_transform(CodeMode())
    elif condition not in ("direct", "oracle"):
        raise ValueError(condition)
    return mcp
