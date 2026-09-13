import sqlite3
import os

DB_PATH = os.path.join("backend", "ai_madac.db")

if not os.path.exists(DB_PATH):
    print(f"Database not found at {DB_PATH}")
    exit(1)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

try:
    cursor.execute("SELECT id, name, email, created_at FROM users")
    rows = cursor.fetchall()
    
    print("\n" + "="*80)
    print(f"{'ID':<34} | {'Name':<25} | {'Email':<30} | {'Created At':<20}")
    print("="*80)
    for row in rows:
        # Format UUID by inserting hyphens for readability if it's hex format
        uid = row[0]
        if len(uid) == 32:
            uid = f"{uid[:8]}-{uid[8:12]}-{uid[12:16]}-{uid[16:20]}-{uid[20:]}"
        print(f"{uid:<34} | {row[1]:<25} | {row[2]:<30} | {row[3]:<20}")
    print("="*80 + "\n")
except sqlite3.OperationalError as e:
    print(f"Error querying users table: {e}")
finally:
    conn.close()
