import subprocess
import sys
import time
import gc
import signal
import requests


class VLLMServerHealthTimeout(Exception):
    """Raised when the vLLM server fails to become healthy within the timeout."""


class VLLMServerManager:
    """Manages a vLLM OpenAI-compatible server as a background subprocess."""

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._port: int | None = None

    def start_server(self, model_id: str, port: int) -> None:
        """Launch the vLLM server without blocking the main thread.

        Parameters
        ----------
        model_id : str
            HuggingFace model identifier (e.g. ``"meta-llama/Llama-2-7b-hf"``).
        port : int
            Port number the server will listen on.
        """
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("A vLLM server is already running. Terminate it first.")

        self._port = port
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_id,
            "--port",
            str(port),
        ]
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def wait_for_health(self, timeout: int = 120) -> None:
        """Poll the ``/health`` endpoint until a 200 OK is received.

        Parameters
        ----------
        timeout : int
            Maximum seconds to wait before raising.

        Raises
        ------
        VLLMServerHealthTimeout
            If the server does not respond with 200 within *timeout* seconds.
        """
        if self._port is None:
            raise RuntimeError("Server has not been started yet.")

        url = f"http://localhost:{self._port}/health"
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(2)

        raise VLLMServerHealthTimeout(
            f"vLLM server did not become healthy within {timeout}s on port {self._port}"
        )

    def terminate(self) -> None:
        """Gracefully stop the server subprocess and release VRAM.

        Sends SIGTERM first, escalates to SIGKILL after 5 seconds,
        then forces a CUDA cache flush and garbage collection.
        """
        if self._process is None or self._process.poll() is not None:
            self._free_vram()
            return

        # -- Graceful shutdown (SIGTERM) -----------------------------------
        self._process.terminate()  # sends SIGTERM (POSIX) / TerminateProcess (Windows)
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # -- Force kill (SIGKILL) --------------------------------------
            self._process.kill()  # sends SIGKILL (POSIX) / TerminateProcess (Windows)
            self._process.wait()

        self._process = None
        self._port = None

        # -- Free GPU memory -----------------------------------------------
        self._free_vram()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _free_vram() -> None:
        """Release VRAM via ``torch.cuda.empty_cache()`` and run garbage collection."""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass  # torch not installed — nothing to clear
