import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from vllm_server_manager import VLLMServerManager


async def test_server():
    manager = VLLMServerManager()
    try:
        manager.start_server("TinyLlama/TinyLlama-1.1B-Chat-v1.0", port=8000)
        await asyncio.to_thread(manager.wait_for_health, 120)
        print("Server is alive and healthy!")
    finally:
        manager.terminate()
        print("Server terminated and VRAM cleared.")


if __name__ == "__main__":
    asyncio.run(test_server())
