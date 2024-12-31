import pickle
import numpy as np
from pathlib import Path


class LMDBStorage:
    def __init__(self, path: Path, readonly=True) -> None:
        self.path = path
        self.db = None
        self._readonly = readonly

    def __contains__(self, key):
        self._connect()
        with self.db.begin() as txn:
            return txn.get(key.encode()) is not None

    def _connect(self):
        import lmdb

        if self.db is None:
            self.db = lmdb.open(
                str(self.path),
                map_size=1024**4,
                subdir=False,
                readonly=self._readonly,
            )

    def __len__(self):
        self._connect()
        with self.db.begin() as txn:
            return txn.stat()["entries"]

    def __getitem__(self, key: str):
        self._connect()
        with self.db.begin() as txn:
            return pickle.loads(txn.get(key.encode()))

    def __setitem__(self, key: str, value):
        self._connect()
        with self.db.begin(write=True) as txn:
            txn.put(key.encode(), pickle.dumps(value))

    def __iter__(self):
        self._connect()
        with self.db.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                yield key.decode()

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None