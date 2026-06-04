"""
Shared LLM utilities: thin wrapper around the local Ollama server.
The system_prompt parameter is the only place scenario/role configuration lives.
"""


def query_ollama(
    prompt: str,
    system_prompt: str,
    model: str,
    base_url: str,
    temperature: float = 0.7,
) -> str | None:
    """
    Send a chat request to a locally running Ollama server.

    Parameters
    ----------
    prompt : str
        The user message to send.
    system_prompt : str
        The system role instructions. Changing the scenario means changing this argument.
    model : str
        Ollama model name, e.g. 'llama3'.
    base_url : str
        Base URL of the Ollama server, e.g. 'http://localhost:11434'.
    temperature : float
        Sampling temperature (0 = deterministic, 1 = creative).

    Returns
    -------
    str or None
        The model's response text, or None if Ollama is unavailable.
    """
    # Import requests first so it is always bound before the except clauses below.
    # If the package is missing, fail fast with a clear message instead of an
    # UnboundLocalError on 'requests.exceptions.ConnectionError'.
    try:
        import requests
    except ImportError:
        print("[WARN] 'requests' package not installed. Run: pip install requests")
        return None

    import json

    try:
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            "options": {"temperature": temperature},
            "stream": False,
        }
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["message"]["content"]

    except requests.exceptions.ConnectionError:
        print("[WARN] Ollama not running. Using rule-based fallback.")
        return None
    except Exception as exc:
        print(f"[WARN] Ollama query failed: {exc}")
        return None
