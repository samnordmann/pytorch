import gc
import unittest
import weakref
from unittest.mock import Mock

import torch
from torch.distributed._transfer import Endpoint
from torch.distributed._transfer._reference import MemoryBackend


class TestTransfer(unittest.TestCase):
    def setUp(self):
        self.backend = MemoryBackend()
        self.a, self.b = Endpoint(self.backend), Endpoint(MemoryBackend())
        self.src = torch.arange(8, dtype=torch.uint8)
        self.dst = torch.zeros_like(self.src)
        self.reg = self.a.register_tensor(self.src)
        self.b.register_tensor(self.dst)
        self.peer = self.a.connect(self.b.metadata())
        self.local = self.a.prepare([(self.src.data_ptr(), 4, 0, 4, 2)], kind="host")
        self.remote = self.a.prepare(
            [(self.dst.data_ptr(), 4, 0, 4, 2)], kind="host", peer=self.peer
        )

    def tearDown(self):
        self.a.close()
        self.b.close()

    def submit(self, operation="WRITE", **kwargs):
        return self.a.submit(
            operation, self.local, [1, 0], self.remote, [0, 1], **kwargs
        )

    def test_indexed_write_and_reuse(self):
        for _ in range(2):
            work = self.submit(notification=b"bucket-7")
            work.wait()
            self.assertEqual(self.dst.tolist(), [4, 5, 6, 7, 0, 1, 2, 3])
            self.assertEqual(self.b.notifications(), {self.backend.name: [b"bucket-7"]})
            self.assertEqual(self.b.notifications(), {})
            work.close()

    def test_read_and_standalone_notification(self):
        self.dst.fill_(9)
        work = self.submit("READ")
        work.wait()
        self.assertEqual(self.src.tolist(), [9] * 8)
        work.close()
        self.a.notify(self.peer, b"ack")
        self.assertEqual(self.b.notifications(), {self.backend.name: [b"ack"]})

    def test_close_in_dependency_order(self):
        work = self.submit()
        for close in (
            self.reg.close,
            self.local.close,
            lambda: self.a.disconnect(self.peer),
        ):
            with self.assertRaises(RuntimeError):
                close()
        work.close()
        self.a.close()
        self.a.close()
        with self.assertRaises(RuntimeError):
            self.a.register_tensor(self.src)

    def test_timeout_and_release_failure_retain_resources(self):
        self.backend.post = Mock(return_value="pending")
        self.backend.poll = Mock(return_value="pending")
        work = self.submit()
        with self.assertRaises(TimeoutError):
            work.wait(timeout=0)
        with self.assertRaises(RuntimeError):
            work.close()
        self.backend.poll.return_value = "done"
        self.backend.release_work = Mock(side_effect=RuntimeError("retry"))
        with self.assertRaises(RuntimeError):
            work.close()
        with self.assertRaises(RuntimeError):
            self.local.close()
        self.backend.release_work.side_effect = None
        work.close()

    def test_post_exception_retains_handle_for_endpoint_cleanup(self):
        self.backend.post = Mock(side_effect=RuntimeError("post uncertain"))
        with self.assertRaises(RuntimeError):
            self.submit()
        with self.assertRaises(RuntimeError):
            self.reg.close()
        self.a.close()  # reference provider reports terminal completion

    def test_tensor_owner_is_retained(self):
        owner = weakref.ref(self.src)
        del self.src
        gc.collect()
        self.assertIsNotNone(owner())
        self.a.close()
        gc.collect()
        self.assertIsNone(owner())

    def test_host_wait_and_invalid_ordering(self):
        event = Mock()
        self.submit(ordering="host_wait", ready_event=event).close()
        event.synchronize.assert_called_once_with()
        for kwargs in (
            {"ordering": "host_wait"},
            {"ordering": "unknown"},
            {"ready_event": event},
        ):
            with self.assertRaises(ValueError):
                self.submit(**kwargs)

    def test_invalid_catalogs_indices_and_registration(self):
        with self.assertRaises(ValueError):
            self.a.prepare([(self.src.data_ptr() + 8, 4, 0)], kind="host")
        with self.assertRaises(ValueError):
            self.a.submit("WRITE", self.remote, [0], self.local, [0])
        with self.assertRaises(ValueError):
            self.a.submit("WRITE", self.local, [2], self.remote, [0])
        with self.assertRaises(ValueError):
            self.a.submit("WRITE", self.local, [], self.remote, [])
        with self.assertRaises(ValueError):
            self.a.register_tensor(torch.zeros(2, 2).t())


if __name__ == "__main__":
    unittest.main()
