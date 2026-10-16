"""Schema parsing. Option order follows the grader: the item's values, then
the gate sentinel if the item is gated."""
from dataclasses import dataclass, field


@dataclass
class Item:
    name: str
    cls: str                      # GIVEN or PREDICT
    values: list
    gated: bool
    sentinel: str
    parent: str | None = None
    observed_if: list = field(default_factory=list)

    @property
    def options(self) -> list:
        return self.values + ([self.sentinel] if self.gated else [])

    @property
    def k_cond(self) -> int:
        return len(self.values)

    @property
    def k_full(self) -> int:
        return len(self.values) + int(self.gated)


def parse_schema(schema: dict):
    """(items, given names, predict names), in schema order."""
    sentinel = schema.get("gated_value", "NA_GATED")
    items: dict[str, Item] = {}
    for name, rec in schema["items"].items():
        if rec["class"] not in ("GIVEN", "PREDICT"):
            continue
        gate = rec.get("gate") or None
        items[name] = Item(
            name=name,
            cls=rec["class"],
            values=list(rec["values"]),
            gated=bool(gate),
            sentinel=sentinel,
            parent=gate["parent"] if gate else None,
            observed_if=list(gate.get("observed_if") or []) if gate else [],
        )
    given = [n for n, it in items.items() if it.cls == "GIVEN"]
    predict = [n for n, it in items.items() if it.cls == "PREDICT"]
    return items, given, predict


def topo_order(items: dict, names: list) -> list:
    """PREDICT items ordered so that gate parents come before their children."""
    order, state = [], {}

    def visit(n):
        if state.get(n):
            return
        state[n] = 1
        p = items[n].parent
        if p is not None and p in items and items[p].cls == "PREDICT":
            visit(p)
        state[n] = 2
        order.append(n)

    for n in names:
        visit(n)
    return order
