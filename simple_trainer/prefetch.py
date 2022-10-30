import time
from threading import Thread, Event
from queue import Queue
from common_pyutil.monitor import Timer


class Prefetcher:
    def __init__(self, dataloader, q_max_size=512):
        self._loader = dataloader
        self._max_size = q_max_size
        self.fetch_ev = Event()
        self.exit_ev = Event()
        self.re_init()

    def re_init(self):
        self._loader_iter = self._loader.__iter__()
        self.q = Queue(self._max_size)
        self.exit_ev.clear()
        self.fetch_ev.clear()
        self._prefetch_thread = Thread(target=self._prefetch)
        self._prefetch_thread.start()
        self.fetched = 0
        self.timer = Timer(True)
        self._finished = False

    def _prefetch(self):
        """The actual prefetch function"""
        try:
            while True and not self.exit_ev.is_set():
                if self.fetch_ev.is_set() and self.q.not_full:
                    with self.timer:
                        self.q.put(self._loader_iter.__next__())
                    self.fetched += 1
                else:
                    time.sleep(.5)
        except StopIteration:
            self._finished = True
            self.finish()

    @property
    def finished(self) -> bool:
        """Is the dataloader iterator finished?"""
        return self._finished

    @property
    def avg_time(self) -> float:
        """Return average time spent fetching the data instances"""
        if self.fetched:
            return self.timer.time / self.fetched
        else:
            return 0

    def start(self):
        """Set the fetch event

        Does NOT clear the exit event. That can only be done with re-
        initialization of the prefetcher.

        """
        self.fetch_ev.set()

    def finish(self):
        """Clear the fetch event and set the exit event"""
        self.fetch_ev.clear()
        self.exit_ev.set()

    def join(self):
        """Join the prefetch thread"""
        self._prefetch_thread.join()

    def get(self):
        """Get item from queue if not empty else raise StopIteration"""
        if self._finished and self.q.empty():
            raise StopIteration
        else:
            return self.q.get()
