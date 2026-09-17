"""Synchronous, in-process CPU backend for examples/tests, NOT networking."""

import ctypes
import uuid
import weakref


class MemoryBackend:
    _agents = weakref.WeakValueDictionary()

    def __init__(self):
        self.name = uuid.uuid4().hex
        self._agents[self.name] = self
        self.regions = {}
        self.messages = {}

    def register(self, regions, kind):
        if kind != "host":
            raise ValueError("reference backend supports CPU memory only")
        handle = object()
        self.regions[handle] = tuple(regions)
        return handle

    def deregister(self, handle):
        del self.regions[handle]

    def metadata(self):
        return self.name.encode()

    def connect(self, metadata):
        name = metadata.decode()
        self._agents[name]  # fail immediately if the peer is absent
        return name

    def disconnect(self, peer):
        pass

    def prepare(self, regions, kind, peer):
        if kind != "host":
            raise ValueError("reference backend supports CPU memory only")
        owner = self if peer is None else self._agents[peer]
        expanded = []
        for row in regions:
            address, length, device = map(int, row[:3])
            stride, count = map(int, row[3:]) if len(row) == 5 else (length, 1)
            if length <= 0 or count <= 0 or stride < length:
                raise ValueError("invalid region")
            for i in range(count):
                span = (address + i * stride, length, device)
                if not any(
                    a <= span[0] and span[0] + length <= a + n and d == device
                    for rows in owner.regions.values()
                    for a, n, d in rows
                ):
                    raise ValueError("descriptor outside registered memory")
                expanded.append(span)
        return owner, expanded

    def release_catalog(self, handle):
        pass

    def create_work(
        self, operation, local, local_indices, remote, remote_indices, notification
    ):
        pairs = []
        for li, ri in zip(local_indices, remote_indices):
            if not (0 <= li < len(local[1]) and 0 <= ri < len(remote[1])):
                raise ValueError("index outside catalog")
            left, right = local[1][li], remote[1][ri]
            if left[1] != right[1]:
                raise ValueError("descriptor lengths differ")
            pairs.append((left, right) if operation == "WRITE" else (right, left))
        return pairs, remote[0], notification

    def post(self, handle):
        pairs, peer, message = handle
        for source, target in pairs:
            ctypes.memmove(target[0], source[0], source[1])
        if message:
            peer.messages.setdefault(self.name, []).append(message)
        return "done"

    def poll(self, handle):
        return "done"

    def release_work(self, handle):
        pass

    def notify(self, peer, message):
        self._agents[peer].messages.setdefault(self.name, []).append(message)

    def notifications(self):
        messages, self.messages = self.messages, {}
        return messages
