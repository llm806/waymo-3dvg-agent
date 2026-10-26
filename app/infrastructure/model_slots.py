"""Bounded provider admission, acquired before a durable call reservation."""
import threading
import time
from contextlib import contextmanager


class ModelSlots:
    def __init__(self, concurrency=2, interval=0.1):
        if not 1 <= concurrency <= 4 or interval < 0:
            raise ValueError('Model concurrency must be 1..4; interval must be nonnegative')
        self.concurrency = concurrency
        self.slots = threading.BoundedSemaphore(concurrency)
        self.lock = threading.Lock()
        self.interval = interval
        self.next_start = 0.0

    def backoff(self, seconds=30):
        with self.lock:
            self.next_start = max(self.next_start, time.monotonic() + seconds)

    def wait_retry(self, heartbeat, check_control):
        """A worker already owns a slot; retries must still honor shared cooldowns."""
        while True:
            check_control()
            heartbeat()
            with self.lock:
                now = time.monotonic()
                if now >= self.next_start:
                    self.next_start = now + self.interval
                    return
            time.sleep(0.05)

    @contextmanager
    def slot(self, heartbeat, check_control):
        acquired = False
        try:
            while not acquired:
                check_control()
                heartbeat()
                with self.lock:
                    if time.monotonic() >= self.next_start:
                        acquired = self.slots.acquire(blocking=False)
                        if acquired:
                            self.next_start = time.monotonic() + self.interval
                if not acquired:
                    time.sleep(0.05)
            check_control()
            yield
        finally:
            if acquired:
                self.slots.release()
