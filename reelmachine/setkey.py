"""Save your Higgsfield API key in Windows Credential Manager.

Run in a normal terminal:  python setkey.py
Paste the key as KEY_ID:KEY_SECRET (it isn't shown while you type).
"""
import getpass

from engine.courier import save_key

key = getpass.getpass("Higgsfield key (KEY_ID:KEY_SECRET): ").strip()
if ":" not in key:
    raise SystemExit("That doesn't look like KEY_ID:KEY_SECRET. Nothing saved.")
save_key(key)
print("Saved to Windows Credential Manager as 'reelmachine-higgsfield'.")
