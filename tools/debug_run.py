#!/usr/bin/env python3
"""Start the `maneuvers` package under debugpy for a VS Code attach session.

Run this script, then use the "Python: Attach to debugpy" configuration in VS Code.
"""
import runpy
import socket
import time

try:
    import debugpy
except Exception:
    raise SystemExit(
        "debugpy is not installed. Install with: pip install -r requirements.txt"
    )

PORT = 5678

def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True

if _port_in_use(PORT):
    print(f"Port {PORT} already in use. Is another debugpy running?")
else:
    print(f"Starting debugpy and listening on 127.0.0.1:{PORT}")
    debugpy.listen(("127.0.0.1", PORT))
    print("Waiting for VS Code to attach... (will block until attached)")
    debugpy.wait_for_client()
    print("Client attached; running maneuvers package.")
    # Run the package as __main__
    runpy.run_module("maneuvers", run_name="__main__")
