import asyncio
import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vllm_server_manager import VLLMServerManager, VLLMServerHealthTimeout


class TestVLLMServerManagerInit(unittest.TestCase):
    """Tests for initial state of VLLMServerManager."""

    def test_initial_state(self):
        manager = VLLMServerManager()
        self.assertIsNone(manager._process)
        self.assertIsNone(manager._port)


class TestStartServer(unittest.TestCase):
    """Tests for the start_server method."""

    @patch("vllm_server_manager.subprocess.Popen")
    def test_start_server_sets_port_and_process(self, mock_popen):
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        manager = VLLMServerManager()
        manager.start_server("TinyLlama/TinyLlama-1.1B-Chat-v1.0", port=8000)

        self.assertEqual(manager._port, 8000)
        self.assertIs(manager._process, mock_proc)
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        self.assertIn("--model", cmd)
        self.assertIn("TinyLlama/TinyLlama-1.1B-Chat-v1.0", cmd)
        self.assertIn("--port", cmd)
        self.assertIn("8000", cmd)

    @patch("vllm_server_manager.subprocess.Popen")
    def test_start_server_raises_if_already_running(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        mock_popen.return_value = mock_proc

        manager = VLLMServerManager()
        manager.start_server("model-a", port=8000)

        with self.assertRaises(RuntimeError):
            manager.start_server("model-b", port=8001)


class TestWaitForHealth(unittest.TestCase):
    """Tests for the wait_for_health method."""

    def test_raises_if_server_not_started(self):
        manager = VLLMServerManager()
        with self.assertRaises(RuntimeError):
            manager.wait_for_health()

    @patch("vllm_server_manager.requests.get")
    def test_returns_on_200(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        manager = VLLMServerManager()
        manager._port = 8000
        manager.wait_for_health(timeout=5)  # should return immediately

    @patch("vllm_server_manager.time.sleep", return_value=None)
    @patch("vllm_server_manager.time.monotonic")
    @patch("vllm_server_manager.requests.get")
    def test_raises_timeout(self, mock_get, mock_monotonic, _mock_sleep):
        import requests

        mock_get.side_effect = requests.ConnectionError
        # First call sets the deadline, subsequent calls exceed it
        mock_monotonic.side_effect = [0, 0, 5, 10]

        manager = VLLMServerManager()
        manager._port = 8000
        with self.assertRaises(VLLMServerHealthTimeout):
            manager.wait_for_health(timeout=3)


class TestTerminate(unittest.TestCase):
    """Tests for the terminate method."""

    @patch.object(VLLMServerManager, "_free_vram")
    def test_terminate_no_process(self, mock_free):
        manager = VLLMServerManager()
        manager.terminate()
        mock_free.assert_called_once()

    @patch.object(VLLMServerManager, "_free_vram")
    def test_terminate_graceful(self, mock_free):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        mock_proc.wait.return_value = 0

        manager = VLLMServerManager()
        manager._process = mock_proc
        manager._port = 8000

        manager.terminate()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_not_called()
        self.assertIsNone(manager._process)
        self.assertIsNone(manager._port)
        mock_free.assert_called_once()

    @patch.object(VLLMServerManager, "_free_vram")
    def test_terminate_force_kill_on_timeout(self, mock_free):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="x", timeout=5), 0]

        manager = VLLMServerManager()
        manager._process = mock_proc
        manager._port = 8000

        manager.terminate()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        mock_free.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Integration smoke test — requires vLLM and a GPU."""

    @unittest.skipUnless(
        # Only run when explicitly requested via TEST_VLLM_INTEGRATION=1
        __import__("os").environ.get("TEST_VLLM_INTEGRATION") == "1",
        "Set TEST_VLLM_INTEGRATION=1 to run live server tests",
    )
    def test_server_lifecycle(self):
        async def _run():
            manager = VLLMServerManager()
            try:
                manager.start_server("TinyLlama/TinyLlama-1.1B-Chat-v1.0", port=8000)
                await asyncio.to_thread(manager.wait_for_health, 120)
                print("Server is alive and healthy!")
            finally:
                manager.terminate()
                print("Server terminated and VRAM cleared.")

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
