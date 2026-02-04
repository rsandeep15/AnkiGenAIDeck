import json
import urllib.error
import urllib.request
from typing import Any, Dict

from config import ANKI_CONNECT_URL


def request(action: str, **params: Any) -> Dict[str, Any]:
    return {"action": action, "params": params, "version": 6}


def invoke(action: str, **params: Any) -> Any:
    request_json = json.dumps(request(action, **params)).encode("utf-8")
    req = urllib.request.Request(ANKI_CONNECT_URL, request_json)
    try:
        with urllib.request.urlopen(req) as response_handle:
            response = json.load(response_handle)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to reach AnkiConnect at {ANKI_CONNECT_URL}: {exc}"
        ) from exc
    if len(response) != 2:
        raise Exception("response has an unexpected number of fields")
    if "error" not in response:
        raise Exception("response is missing required error field")
    if "result" not in response:
        raise Exception("response is missing required result field")
    if response["error"] is not None:
        raise Exception(response["error"])
    return response["result"]
