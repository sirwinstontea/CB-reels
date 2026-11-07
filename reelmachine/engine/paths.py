"""Where things live. DATA = read-only files shipped with the app (markers, window);
HOME = your folder for jobs, settings, desk pictures, sounds, hand references.
Running from source both are the project folder; as an .exe, HOME is the folder holding the .exe."""
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    DATA = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    HOME = Path(sys.executable).parent
else:
    DATA = HOME = Path(__file__).resolve().parents[1]
