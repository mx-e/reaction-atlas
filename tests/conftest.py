"""Shared test fixtures for comprehensive verification.

Works on macOS with PostgreSQL (Homebrew) or SQLite fallback.
No Docker, no GPU required.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
