"""SQLite-backed order repository used to persist trade activity safely."""
import sqlite3
import threading
import time
from pathlib import Path

try:
    from paths import resolve_data_path
except ImportError:
    resolve_data_path = None


class OrderStore:
    """Thread-safe helper that persists order information in SQLite."""

    def __init__(self, db_name="orders.db"):
        """Resolve the database path and ensure the schema exists."""
        if resolve_data_path:
            self.db_name = str(resolve_data_path(Path(db_name).name))
        else:
            self.db_name = db_name
        self.local = threading.local()
        self._initialize_db()

    def get_conn(self):
        """Return the thread-local SQLite connection, creating it on demand."""
        if not hasattr(self.local, "conn"):
            self.local.conn = sqlite3.connect(self.db_name, timeout=10)
            self.local.conn.execute("PRAGMA journal_mode=WAL;")
            self.local.conn.execute("PRAGMA synchronous=NORMAL;")
            self.local.conn.execute("PRAGMA busy_timeout=5000;")
        return self.local.conn

    def _initialize_db(self):
        """Create the ``orders`` table if it does not already exist."""
        conn = self.get_conn()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    status TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

    def add_order(self, order_id, symbol, side, price, amount, status):
        """Insert an order row, retrying automatically if the DB is locked."""
        conn = self.get_conn()
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO orders (order_id, symbol, side, price, amount, status)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (order_id, symbol, side, price, amount, status),
                    )
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    if attempt < max_retries - 1:
                        time.sleep(0.1)
                        continue
                    else:
                        print(f"DB Error on add_order after retries: {e}")
                else:
                    print(f"DB Error on add_order: {e}")
                    break
            except sqlite3.Error as e:
                print(f"DB Error on add_order: {e}")
                break

    def update_order_status(self, order_id, status):
        """Update an order's status while handling potential lock contention."""
        conn = self.get_conn()
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with conn:
                    conn.execute(
                        "UPDATE orders SET status = ? WHERE order_id = ?",
                        (status, order_id),
                    )
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    if attempt < max_retries - 1:
                        time.sleep(0.1)
                        continue
                    else:
                        print(f"DB Error on update_order_status after retries: {e}")
                else:
                    print(f"DB Error on update_order_status: {e}")
                    break
            except sqlite3.Error as e:
                print(f"DB Error on update_order_status: {e}")
                break

    def get_pending_orders(self):
        """Return all orders that are still marked as pending."""
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT order_id, symbol FROM orders WHERE status = 'pending'")
        rows = cursor.fetchall()
        return [{"order_id": row[0], "symbol": row[1]} for row in rows]

    def close(self):
        """Close and remove the thread-local SQLite connection if present."""
        if hasattr(self.local, "conn"):
            self.local.conn.close()
            del self.local.conn
