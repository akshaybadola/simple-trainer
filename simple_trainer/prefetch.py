import time
from threading import Thread, Event
from queue import Queue
from common_pyutil.monitor import Timer


class Prefetcher:
    def __init__(self, dataloader, q_max_size=512):
        self.fetch_ev = Event()
        self.exit_ev = Event()
        self.loader = dataloader
        self._loader_iter = self.loader.__iter__()
        self.q = Queue(q_max_size)
        self.exit_ev.clear()
        self.fetch_ev.clear()
        self.prefetch_thread = Thread(target=self._prefetch)
        self.prefetch_thread.start()
        self.fetched = 0
        self.timer = Timer(True)

    def _prefetch(self):
        while True and not self.exit_ev.is_set():
            if self.fetch_ev.is_set() and self.q.not_full:
                with self.timer:
                    self.q.put(self._loader_iter.__next__())
                self.fetched += 1
            else:
                time.sleep(.5)

    @property
    def avg_time(self) -> float:
        if self.fetched:
            return self.timer.time / self.fetched
        else:
            return 0

    def start(self):
        self.fetch_ev.set()

    def join(self):
        self.fetch_ev.clear()
        self.exit_ev.set()

    def get(self):
        return self.q.get()
