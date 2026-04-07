import sys
from pathlib import Path

# Ensure leakhunter/ is on sys.path so bare imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
