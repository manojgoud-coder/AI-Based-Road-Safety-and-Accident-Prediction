import sqlite3

conn = sqlite3.connect("users.db")
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cur.fetchall()

print("Tables:")
for t in tables:
    print(t)

print("\nUsers Table:")
cur.execute("SELECT * FROM users")
for row in cur.fetchall():
    print(row)

print("\nPrediction History:")
cur.execute("SELECT * FROM prediction_history")
for row in cur.fetchall():
    print(row)

conn.close()