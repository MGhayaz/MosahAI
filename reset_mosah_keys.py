import os
import sqlite3


def main():
    db_path = os.path.join(os.path.dirname(__file__), "usage_tracker.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE api_keys_health
        SET status='Active',
            usage_count=0,
            error_type=NULL
        """
    )

    conn.commit()
    conn.close()

    print("All API keys reset to Active.")


if __name__ == "__main__":
    main()
