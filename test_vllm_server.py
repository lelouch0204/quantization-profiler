import asyncio
from vllm_server_manager import VLLMServerManager


async def test_vllm_server() -> None:
    """Start a vLLM server, verify it's healthy, then shut it down."""
    manager = VLLMServerManager()

    try:
        manager.start_server(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            port=8000,
        )
        manager.wait_for_health(timeout=120)
        print("Server is alive and healthy!")
    finally:
        manager.terminate()
        print("Server terminated and VRAM cleared.")


if __name__ == "__main__":
    asyncio.run(test_vllm_server())
