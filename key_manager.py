class KeyManager:
    """Select least-used active keys and update health status via tracker."""

    def __init__(self, tracker):
        self.tracker = tracker

    def get_next_key(self):
        return self.tracker.get_active_least_used_key()

    def mark_success(self, api_key):
        self.tracker.mark_success(api_key)

    def mark_error(self, api_key, status_code):
        if api_key and status_code is not None:
            self.tracker.mark_error(api_key, status_code)

    def health_snapshot(self):
        return self.tracker.list_health()
