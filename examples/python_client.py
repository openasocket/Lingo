#!/usr/bin/env python3
"""
Example: Using lingo from Python via the HTTP server.

1. Start the server:
   cargo run --example server --features server,metal

2. Run this script:
   python examples/python_client.py
"""

import json
import urllib.request

SERVER_URL = "http://localhost:3000"


def translate(text: str, source: str = "en", target: str = "fr") -> dict:
    """Translate text using the lingo server."""
    data = json.dumps({
        "text": text,
        "source": source,
        "target": target,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{SERVER_URL}/translate",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    # Single translation
    result = translate("Hello, how are you?", "en", "fr")
    print(f"French: {result['translation']} ({result['duration_ms']}ms)")

    # Multiple languages
    text = "The weather is beautiful today."
    for lang in ["es", "ja", "ar", "ko", "de", "zh", "ru"]:
        result = translate(text, "en", lang)
        print(f"[{lang}] {result['translation']} ({result['duration_ms']}ms)")


if __name__ == "__main__":
    main()
