import random
import sqlite3
from datetime import datetime, timedelta, timezone


class CircuitBreakerOpenError(RuntimeError):
    """Raised when the global circuit breaker is open and requests must pause."""

    def __init__(self, wait_seconds):
        self.wait_seconds = max(1, int(wait_seconds))
        super().__init__(
            f"Circuit breaker is open. Wait {self.wait_seconds}s before sending a new request."
        )


class APIHealthTracker:
    """SQLite-backed tracker for API key usage and health."""

    def __init__(
        self,
        db_path,
        keys,
        cooldown_429_hours=24,
        cooldown_404_minutes=15,
        circuit_breaker_failure_threshold=3,
        circuit_breaker_cooldown_seconds=120,
        smart_cooldown_min_minutes=30,
        smart_cooldown_max_minutes=44,
    ):
        self.db_path = db_path
        self.cooldown_429_hours = max(1, int(cooldown_429_hours))
        self.cooldown_404_minutes = max(1, int(cooldown_404_minutes))
        self.circuit_breaker_failure_threshold = max(1, int(circuit_breaker_failure_threshold))
        self.circuit_breaker_cooldown_seconds = max(1, int(circuit_breaker_cooldown_seconds))
        self.smart_cooldown_min_minutes = int(smart_cooldown_min_minutes)
        self.smart_cooldown_max_minutes = int(smart_cooldown_max_minutes)
        if self.smart_cooldown_min_minutes > self.smart_cooldown_max_minutes:
            self.smart_cooldown_min_minutes, self.smart_cooldown_max_minutes = (
                self.smart_cooldown_max_minutes,
                self.smart_cooldown_min_minutes,
            )

        self.consecutive_failures = 0
        self.circuit_open_until = None
        self.previous_key_cooldown = None
        self._ensure_schema()
        self.sync_keys(keys)

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_key TEXT UNIQUE NOT NULL,
                    usage_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'Active',
                    last_used TEXT,
                    error_type TEXT,
                    cooldown_until TEXT
                )
                """
            )
            existing_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(api_keys_health)").fetchall()
            }
            if "cooldown_until" not in existing_columns:
                conn.execute("ALTER TABLE api_keys_health ADD COLUMN cooldown_until TEXT")
            conn.commit()

    def _now(self):
        return datetime.now(timezone.utc)

    def _now_iso(self):
        return self._now().isoformat()

    def _parse_iso(self, value):
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(str(value))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _parse_error_count(self, error_text, code):
        if not error_text:
            return 0
        base = str(code)
        raw = str(error_text).strip().lower()
        if raw == base.lower():
            return 1
        prefix = f"{base.lower()}x"
        if raw.startswith(prefix):
            suffix = raw[len(prefix) :]
            if suffix.isdigit():
                return int(suffix)
        return 0

    def _format_error_count(self, code, count):
        if count <= 1:
            return str(code)
        return f"{code}x{count}"

    def _next_unique_cooldown_minutes(self):
        current_key_cooldown = random.randint(
            self.smart_cooldown_min_minutes, self.smart_cooldown_max_minutes
        )
        if self.previous_key_cooldown == current_key_cooldown:
            possible_values = [
                value
                for value in range(self.smart_cooldown_min_minutes, self.smart_cooldown_max_minutes + 1)
                if value != self.previous_key_cooldown
            ]
            if possible_values:
                current_key_cooldown = random.choice(possible_values)
        self.previous_key_cooldown = current_key_cooldown
        return current_key_cooldown

    def _set_circuit_open(self):
        self.circuit_open_until = self._now() + timedelta(seconds=self.circuit_breaker_cooldown_seconds)
        self.consecutive_failures = 0

    def _register_failure(self):
        wait_seconds = self.get_circuit_wait_seconds()
        if wait_seconds > 0:
            return

        self.consecutive_failures += 1
        if self.consecutive_failures >= self.circuit_breaker_failure_threshold:
            self._set_circuit_open()

    def get_circuit_wait_seconds(self):
        if not self.circuit_open_until:
            return 0
        remaining_seconds = (self.circuit_open_until - self._now()).total_seconds()
        if remaining_seconds <= 0:
            self.circuit_open_until = None
            return 0
        return int(remaining_seconds + 0.999)

    def _raise_if_circuit_open(self):
        wait_seconds = self.get_circuit_wait_seconds()
        if wait_seconds > 0:
            raise CircuitBreakerOpenError(wait_seconds=wait_seconds)

    def sync_keys(self, keys):
        active_keys = [str(key).strip() for key in keys if str(key).strip()]
        if not active_keys:
            raise ValueError("No API keys supplied for tracker sync.")

        with self._connect() as conn:
            existing_rows = conn.execute("SELECT api_key FROM api_keys_health").fetchall()
            existing_keys = {row["api_key"] for row in existing_rows}
            target_keys = set(active_keys)

            for key in active_keys:
                if key not in existing_keys:
                    conn.execute(
                        """
                        INSERT INTO api_keys_health
                        (api_key, usage_count, status, last_used, error_type, cooldown_until)
                        VALUES (?, 0, 'Active', NULL, NULL, NULL)
                        """,
                        (key,),
                    )

            for stale_key in sorted(existing_keys - target_keys):
                conn.execute("DELETE FROM api_keys_health WHERE api_key = ?", (stale_key,))

            conn.commit()

    def _reactivate_expired_cooldowns(self):
        now = self._now()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, status, last_used, error_type, cooldown_until
                FROM api_keys_health
                WHERE status = 'Cooldown'
                """
            ).fetchall()

            for row in rows:
                cooldown_until = self._parse_iso(row["cooldown_until"])
                if cooldown_until:
                    if now >= cooldown_until:
                        conn.execute(
                            """
                            UPDATE api_keys_health
                            SET status = 'Active', error_type = NULL, cooldown_until = NULL
                            WHERE id = ?
                            """,
                            (row["id"],),
                        )
                    continue

                last_used = self._parse_iso(row["last_used"])
                if not last_used:
                    conn.execute(
                        """
                        UPDATE api_keys_health
                        SET status = 'Active', error_type = NULL, cooldown_until = NULL
                        WHERE id = ?
                        """,
                        (row["id"],),
                    )
                    continue

                error_text = str(row["error_type"] or "").lower()
                if error_text.startswith("429"):
                    if now - last_used >= timedelta(hours=self.cooldown_429_hours):
                        conn.execute(
                            """
                            UPDATE api_keys_health
                            SET status = 'Active', error_type = NULL, cooldown_until = NULL
                            WHERE id = ?
                            """,
                            (row["id"],),
                        )
                elif error_text.startswith("404"):
                    if now - last_used >= timedelta(minutes=self.cooldown_404_minutes):
                        conn.execute(
                            """
                            UPDATE api_keys_health
                            SET status = 'Active', error_type = NULL, cooldown_until = NULL
                            WHERE id = ?
                            """,
                            (row["id"],),
                        )
                elif now - last_used >= timedelta(minutes=5):
                    conn.execute(
                        """
                        UPDATE api_keys_health
                        SET status = 'Active', error_type = NULL, cooldown_until = NULL
                        WHERE id = ?
                        """,
                        (row["id"],),
                    )

            conn.commit()

    def get_active_least_used_key(self):
        self._raise_if_circuit_open()
        self._reactivate_expired_cooldowns()

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, api_key, usage_count, status
                FROM api_keys_health
                WHERE status = 'Active'
                ORDER BY usage_count ASC, COALESCE(last_used, '') ASC, id ASC
                LIMIT 1
                """
            ).fetchone()

            if not row:
                raise RuntimeError("No active API keys available. All keys are cooling down or blocked.")

            conn.execute(
                """
                UPDATE api_keys_health
                SET usage_count = usage_count + 1, last_used = ?, error_type = NULL, cooldown_until = NULL
                WHERE id = ?
                """,
                (self._now_iso(), row["id"]),
            )
            conn.commit()
            return row["api_key"]

    def mark_success(self, api_key):
        self.consecutive_failures = 0
        self.circuit_open_until = None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE api_keys_health
                SET status = CASE WHEN status = 'Blocked' THEN status ELSE 'Active' END,
                    last_used = ?,
                    error_type = NULL,
                    cooldown_until = NULL
                WHERE api_key = ?
                """,
                (self._now_iso(), api_key),
            )
            conn.commit()

    def mark_error(self, api_key, error_code):
        error_code = int(error_code) if str(error_code).isdigit() else None
        if error_code is None:
            return

        self._register_failure()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status, error_type FROM api_keys_health WHERE api_key = ?",
                (api_key,),
            ).fetchone()
            if not row:
                return

            previous_error = row["error_type"]
            now = self._now()
            now_iso = now.isoformat()
            cooldown_until = None

            if error_code in {429, 503}:
                status = "Cooldown"
                current_key_cooldown = self._next_unique_cooldown_minutes()
                cooldown_until = (now + timedelta(minutes=current_key_cooldown)).isoformat()
                error_type = f"{error_code}|cdm={current_key_cooldown}"
            elif error_code == 404:
                count = self._parse_error_count(previous_error, 404) + 1
                status = "Blocked" if count >= 5 else "Cooldown"
                error_type = self._format_error_count(404, count)
                if status == "Cooldown":
                    cooldown_until = (now + timedelta(minutes=self.cooldown_404_minutes)).isoformat()
            else:
                status = "Cooldown"
                error_type = str(error_code)
                cooldown_until = (now + timedelta(minutes=5)).isoformat()

            conn.execute(
                """
                UPDATE api_keys_health
                SET status = ?, last_used = ?, error_type = ?, cooldown_until = ?
                WHERE api_key = ?
                """,
                (status, now_iso, error_type, cooldown_until, api_key),
            )
            conn.commit()

    def list_health(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, api_key, usage_count, status, last_used, error_type, cooldown_until
                FROM api_keys_health
                ORDER BY id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def reset_all_keys(self, reset_usage_count=True):
        """Emergency utility: clear cooldown/blocked state for all keys."""
        now_iso = self._now_iso()
        with self._connect() as conn:
            if reset_usage_count:
                cursor = conn.execute(
                    """
                    UPDATE api_keys_health
                    SET status = 'Active',
                        error_type = NULL,
                        cooldown_until = NULL,
                        usage_count = 0,
                        last_used = ?
                    """,
                    (now_iso,),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE api_keys_health
                    SET status = 'Active',
                        error_type = NULL,
                        cooldown_until = NULL,
                        last_used = ?
                    """,
                    (now_iso,),
                )
            conn.commit()
            self.consecutive_failures = 0
            self.circuit_open_until = None
            return cursor.rowcount
