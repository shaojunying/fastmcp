"""Benchmark tasks, stratified by tool-call depth.

``depth`` is the number of distinct tools on the longest dependent chain.
``calls`` is the number of real tool calls in the reference solution
(fan-out makes it larger than depth).

``solution`` is the reference plan as a Python body. ``@name`` marks a tool
reference. The scripted model resolves each ``@name`` to whatever tool it
discovered, so a discovery miss changes which tool actually runs. Real
models ignore ``solution``; they only get ``prompt``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TOOL_REF = r"(?<=call_tool\()@(\w+)"  # "@name" right after "call_tool(" only


@dataclass(frozen=True)
class Task:
    id: str
    depth: int
    prompt: str
    expected: str
    solution: str

    @property
    def needed(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(re.findall(TOOL_REF, self.solution)))


TASKS = [
    # ---- depth 1 ----------------------------------------------------------
    Task(
        "t01",
        1,
        "What loyalty tier is the customer with email ana@example.com in? Answer with the tier name only.",
        "gold",
        'c = await call_tool(@find_customer, {"email": "ana@example.com"})\nreturn c["tier"]',
    ),
    Task(
        "t02",
        1,
        "How many units of SKU-1042 are on hand in the eu-west warehouse? Answer with a number.",
        "23",
        'r = await call_tool(@check_inventory, {"sku": "SKU-1042", "warehouse": "eu-west"})\nreturn r["on_hand"]',
    ),
    Task(
        "t03",
        1,
        "What is the estimated arrival date of order O-5001? Answer as YYYY-MM-DD.",
        "2026-11-25",
        'r = await call_tool(@track_shipment, {"order_id": "O-5001"})\nreturn r["eta"]',
    ),
    Task(
        "t04",
        1,
        "Convert 12345 US cents to euros. Answer with the amount in EUR, 2 decimals.",
        "113.57",
        'r = await call_tool(@convert_currency, {"amount_cents": 12345, "from_currency": "USD", "to_currency": "EUR"})\nreturn r["amount"]',
    ),
    # ---- depth 2 ----------------------------------------------------------
    Task(
        "t05",
        2,
        "How many open orders does the customer bo@example.com have? Answer with a number.",
        "2",
        'c = await call_tool(@find_customer, {"email": "bo@example.com"})\n'
        'r = await call_tool(@list_orders, {"customer_id": c["customer_id"], "status": "open"})\n'
        'return len(r["orders"])',
    ),
    Task(
        "t06",
        2,
        "What is the name of the product in order O-5003? Answer with the product name.",
        "Desk Lamp",
        'o = await call_tool(@get_order, {"order_id": "O-5003"})\n'
        'p = await call_tool(@get_product, {"sku": o["items"][0]["sku"]})\n'
        'return p["name"]',
    ),
    Task(
        "t07",
        2,
        "How many days does the customer bo@example.com have to request a refund? Answer with a number.",
        "30",
        'c = await call_tool(@find_customer, {"email": "bo@example.com"})\n'
        'r = await call_tool(@get_refund_policy, {"tier": c["tier"]})\n'
        'return r["window_days"]',
    ),
    Task(
        "t08",
        2,
        "What is the total of order O-5002 in Japanese yen? Answer with the amount, 2 decimals.",
        "14520.26",
        'o = await call_tool(@get_order, {"order_id": "O-5002"})\n'
        'r = await call_tool(@convert_currency, {"amount_cents": o["total_cents"], "from_currency": o["currency"], "to_currency": "JPY"})\n'
        'return r["amount"]',
    ),
    # ---- depth 3 ----------------------------------------------------------
    Task(
        "t09",
        3,
        "Which carrier is delivering the most recently placed shipped order of ana@example.com? Answer with the carrier name.",
        "DHL",
        'c = await call_tool(@find_customer, {"email": "ana@example.com"})\n'
        'r = await call_tool(@list_orders, {"customer_id": c["customer_id"], "status": "shipped"})\n'
        'latest = sorted(r["orders"], key=lambda o: o["placed_on"])[-1]\n'
        's = await call_tool(@track_shipment, {"order_id": latest["order_id"]})\n'
        'return s["carrier"]',
    ),
    Task(
        "t10",
        3,
        "What is the combined total in EUR of all open orders of bo@example.com? Answer with the amount, 2 decimals.",
        "95.63",
        'c = await call_tool(@find_customer, {"email": "bo@example.com"})\n'
        'r = await call_tool(@list_orders, {"customer_id": c["customer_id"], "status": "open"})\n'
        'total = sum(o["total_cents"] for o in r["orders"])\n'
        'x = await call_tool(@convert_currency, {"amount_cents": total, "from_currency": "USD", "to_currency": "EUR"})\n'
        'return x["amount"]',
    ),
    Task(
        "t11",
        3,
        "What restocking fee percentage applies to the customer who placed order O-5003? Answer with a number.",
        "5",
        'o = await call_tool(@get_order, {"order_id": "O-5003"})\n'
        'c = await call_tool(@get_customer, {"customer_id": o["customer_id"]})\n'
        'r = await call_tool(@get_refund_policy, {"tier": c["tier"]})\n'
        'return r["restocking_fee_pct"]',
    ),
    Task(
        "t12",
        3,
        "What is the sum, in EUR, of the unit list prices of the distinct products in order O-5002? Answer with the amount, 2 decimals.",
        "76.34",
        'o = await call_tool(@get_order, {"order_id": "O-5002"})\n'
        "total = 0\n"
        'for it in o["items"]:\n'
        '    p = await call_tool(@get_product, {"sku": it["sku"]})\n'
        '    total += p["price_cents"]\n'
        'x = await call_tool(@convert_currency, {"amount_cents": total, "from_currency": "USD", "to_currency": "EUR"})\n'
        'return x["amount"]',
    ),
    # ---- depth 4 ----------------------------------------------------------
    Task(
        "t13",
        4,
        "Are all items in the open orders of cy@example.com in stock (on hand > 0) at the ap-south warehouse? Answer yes or no.",
        "no",
        'c = await call_tool(@find_customer, {"email": "cy@example.com"})\n'
        'r = await call_tool(@list_orders, {"customer_id": c["customer_id"], "status": "open"})\n'
        "ok = True\n"
        'for row in r["orders"]:\n'
        '    o = await call_tool(@get_order, {"order_id": row["order_id"]})\n'
        '    for it in o["items"]:\n'
        '        s = await call_tool(@check_inventory, {"sku": it["sku"], "warehouse": "ap-south"})\n'
        '        if s["on_hand"] <= 0:\n'
        "            ok = False\n"
        'return "yes" if ok else "no"',
    ),
    Task(
        "t14",
        4,
        "Across the delivered orders of ana@example.com, which product category had the highest spend (unit price x quantity)? Answer with the category.",
        "kitchen",
        'c = await call_tool(@find_customer, {"email": "ana@example.com"})\n'
        'r = await call_tool(@list_orders, {"customer_id": c["customer_id"], "status": "delivered"})\n'
        "spend = {}\n"
        'for row in r["orders"]:\n'
        '    o = await call_tool(@get_order, {"order_id": row["order_id"]})\n'
        '    for it in o["items"]:\n'
        '        p = await call_tool(@get_product, {"sku": it["sku"]})\n'
        '        spend[p["category"]] = spend.get(p["category"], 0) + p["price_cents"] * it["qty"]\n'
        "return sorted(spend.items(), key=lambda kv: kv[1])[-1][0]",
    ),
    Task(
        "t15",
        4,
        "How many units, in total, are on hand at the eu-west warehouse for all SKUs in the open orders of bo@example.com? Count each distinct SKU once. Answer with a number.",
        "45",
        'c = await call_tool(@find_customer, {"email": "bo@example.com"})\n'
        'r = await call_tool(@list_orders, {"customer_id": c["customer_id"], "status": "open"})\n'
        "skus = []\n"
        'for row in r["orders"]:\n'
        '    o = await call_tool(@get_order, {"order_id": row["order_id"]})\n'
        '    for it in o["items"]:\n'
        '        if it["sku"] not in skus:\n'
        '            skus.append(it["sku"])\n'
        "total = 0\n"
        "for sku in skus:\n"
        '    s = await call_tool(@check_inventory, {"sku": sku, "warehouse": "eu-west"})\n'
        '    total += s["on_hand"]\n'
        "return total",
    ),
    Task(
        "t16",
        4,
        "Find the customer who placed order O-5007. How many units are on hand at us-east for all items in that customer's open orders? Answer with a number.",
        "7",
        'o = await call_tool(@get_order, {"order_id": "O-5007"})\n'
        'r = await call_tool(@list_orders, {"customer_id": o["customer_id"], "status": "open"})\n'
        "total = 0\n"
        'for row in r["orders"]:\n'
        '    oo = await call_tool(@get_order, {"order_id": row["order_id"]})\n'
        '    for it in oo["items"]:\n'
        '        s = await call_tool(@check_inventory, {"sku": it["sku"], "warehouse": "us-east"})\n'
        '        total += s["on_hand"]\n'
        "return total",
    ),
]

# What a model would plausibly type when it searches for each capability.
SEARCH_QUERY = {
    "find_customer": "find customer by email",
    "get_customer": "get customer by id",
    "list_orders": "list orders of a customer by status",
    "get_order": "get order details line items",
    "get_product": "get product details by sku",
    "check_inventory": "check inventory units on hand in warehouse",
    "track_shipment": "track shipment carrier eta",
    "convert_currency": "convert currency amount",
    "get_refund_policy": "refund policy for loyalty tier",
}
SEARCH_REGEX = {
    "find_customer": "customer.*email",
    "get_customer": "customer.*customer_id",
    "list_orders": "list.*orders",
    "get_order": "order.*items",
    "get_product": "product.*sku",
    "check_inventory": "inventory|on hand",
    "track_shipment": "shipment|track",
    "convert_currency": "convert.*currenc",
    "get_refund_policy": "refund",
}


def check_answer(answer: str | None, expected: str) -> bool:
    if answer is None:
        return False
    a = answer.strip().strip('"').strip().lower()
    e = expected.strip().lower()
    try:
        return abs(float(a) - float(e)) <= 0.011
    except ValueError:
        return a == e
