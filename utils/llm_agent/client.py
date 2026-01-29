import os
from tabnanny import verbose
import requests
import json
import time
import logging
from typing import Optional, Dict, Any, List, Union

DEFAULT_TIMEOUT = 120

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VLLM_Client")


class VLLMClient:
    """
    vLLM Service Client.
    Wrapper for HTTP calls compatible with vLLM/OpenAI interfaces.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 160,
        verbose: bool = False,
    ):
        """
        Initialize the client.
        :param verbose: If True, prints detailed raw responses for debugging.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose
        self.session = requests.Session()

        # Set common headers
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "VLLM-Client-Refined/1.1",
            }
        )

        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

        logger.info(
            f"Client initialized. Target: {self.base_url} | Verbose: {self.verbose}"
        )

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Internal generic request handler."""
        url = f"{self.base_url}{endpoint}"
        current_timeout = kwargs.pop("timeout", self.timeout)

        try:
            response = self.session.request(
                method, url, timeout=current_timeout, **kwargs
            )

            # --- [Verbose Switch Logic] ---
            if self.verbose:
                print(f"\n----------- [DEBUG] Endpoint: {endpoint} -----------")
                print(f"Status Code: {response.status_code}")
                print("Raw Response Text:")
                print(response.text)
                print("--------------------------------------------------\n")
            # ------------------------------

            response.raise_for_status()

            if response.status_code == 204:
                return {}

            return response.json()

        except requests.exceptions.JSONDecodeError:
            logger.error("Failed to parse JSON.")
            if not self.verbose:
                logger.error(f"Raw invalid response: {response.text}")
            return {"error": "Invalid JSON response", "raw": response.text}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed [{endpoint}]: {e}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Check basic service health."""
        return self._request("GET", "/health", timeout=5)

    def list_models(self) -> List[str]:
        """Get list of available model IDs."""
        data = self._request("GET", "/v1/models", timeout=10)
        if isinstance(data, dict) and "data" in data:
            return [m["id"] for m in data["data"]]
        return []

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        reasoning_effort: str = "low",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request.
        :param reasoning_effort: 'low', 'medium', or 'high'
        """
        if not model:
            models = self.list_models()
            if models:
                model = models[0]
                logger.info(f"No model specified. Auto-selected: {model}")
            else:
                return {"error": "No models available"}

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,  # Added parameter
            **kwargs,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        logger.info(
            f"Sending request -> Model: {model} | Reasoning: {reasoning_effort}"
        )

        start_time = time.time()
        response = self._request("POST", "/v1/chat/completions", json=payload)
        duration = time.time() - start_time

        if "error" not in response:
            usage = response.get("usage", {})
            logger.info(
                f"Success ({duration:.2f}s) | Tokens Used: {usage.get('total_tokens', 'N/A')}"
            )

        return response


def main():
    """Test Entry Point"""

    # ================= USER CONFIGURATION =================
    # 1. Output Control: True = Show raw JSON/Headers, False = Clean output
    SHOW_DEBUG_INFO = True
    DEFAULT_BASE_URL = ""
    # 2. Reasoning Strength: 'low', 'medium', 'high'
    REASONING_LEVEL = "medium"
    # ======================================================

    client = VLLMClient(base_url=DEFAULT_BASE_URL)

    # 1. Test Connection
    print(f"Testing connection to {client.base_url}...")
    client.health_check()

    # 2. Get Models
    models = client.list_models()

    if not models:
        print("No models available. Aborting.")
        return

    # 3. Test Single Chat with Reasoning Config
    print(f"\n--- Testing Single Chat (Reasoning: {REASONING_LEVEL}) ---")

    resp = client.chat_completion(
        messages=[{"role": "user", "content": "Explain 1+1=2."}],
        model=models[0],
        max_tokens=2000,
        reasoning_effort=REASONING_LEVEL,
    )

    if "choices" in resp:
        print(f"Response\n: {resp['choices'][0]['message']['content']}")
    else:
        print(f"Error: {resp}")


if __name__ == "__main__":
    main()
