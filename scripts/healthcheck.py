#!/usr/bin/env python3
"""Container health: is the SIP registration alive?

Writes nothing; the agent touches a heartbeat file while registered, so a stale
file means registration has been down longer than the refresh interval.
"""

import os
import sys
import time

HEARTBEAT = os.environ.get("HELPDESK_HEARTBEAT", "/tmp/helpdesk-registered")
MAX_AGE = int(os.environ.get("HELPDESK_HEARTBEAT_MAX_AGE", "120"))

if not os.path.exists(HEARTBEAT):
    print(f"no heartbeat at {HEARTBEAT}", file=sys.stderr)
    sys.exit(1)

age = time.time() - os.path.getmtime(HEARTBEAT)
if age > MAX_AGE:
    print(f"heartbeat is {age:.0f}s old (max {MAX_AGE}s): registration lost", file=sys.stderr)
    sys.exit(1)

print(f"registered, heartbeat {age:.0f}s old")
