#!/usr/bin/env python3
import csv
import sys
with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))
assert rows, "No frame records"
def total(key):
    return sum(int(float(row.get(key) or 0)) for row in rows if row.get(key) not in ("NA", "nan"))
for key in ("physical_group_size_3", "physical_residual_created", "physical_residual_accepted", "joint_accepted"):
    value = total(key)
    print(f"{key}={value}")
    assert value > 0, f"{key} was never active"
assert all(row.get("virtual_views") == "3" for row in rows)
assert any(row.get("tracking_state") == "2" for row in rows)
print("ONLINE_THREE_VIEW_OPTIMIZATION_PASS")
