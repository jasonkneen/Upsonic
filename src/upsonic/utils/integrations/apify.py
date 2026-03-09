"""Apify integration utility helpers."""

from __future__ import annotations

import string
from typing import Any, Dict, List, Optional, Tuple

import requests


# Constants
MAX_DESCRIPTION_LEN = 350
REQUESTS_TIMEOUT_SECS = 300
APIFY_API_ENDPOINT_GET_DEFAULT_BUILD = "https://api.apify.com/v2/acts/{actor_id}/builds/default"


def create_apify_client(token: str):
    """Create an Apify client instance with a custom user-agent.

    Args:
        token: Apify API token.

    Returns:
        An ApifyClient instance.
    """
    from apify_client import ApifyClient

    client = ApifyClient(token)
    if http_client := getattr(client.http_client, "httpx_client", None):
        http_client.headers["user-agent"] += "; Origin/upsonic"
    return client


def actor_id_to_tool_name(actor_id: str) -> str:
    """Turn actor_id into a valid tool/method name.

    Args:
        actor_id: Actor ID from Apify store (e.g. 'apify/web-scraper').

    Returns:
        A valid Python identifier for use as a tool name.
    """
    valid_chars = string.ascii_letters + string.digits + "_"
    return "apify_actor_" + "".join(char if char in valid_chars else "_" for char in actor_id)


def get_actor_latest_build(apify_client, actor_id: str) -> Dict[str, Any]:
    """Get the latest build of an Actor from the default build tag.

    Args:
        apify_client: An ApifyClient instance.
        actor_id: Actor ID from Apify store.

    Returns:
        The latest build data dict of the Actor.
    """
    actor = apify_client.actor(actor_id).get()
    if not actor:
        raise ValueError(f"Actor {actor_id} not found.")

    actor_obj_id = actor.get("id")
    if not actor_obj_id:
        raise ValueError(f"Failed to get the Actor object ID for {actor_id}.")

    url = APIFY_API_ENDPOINT_GET_DEFAULT_BUILD.format(actor_id=actor_obj_id)
    response = requests.request("GET", url, timeout=REQUESTS_TIMEOUT_SECS)

    build = response.json()
    if not isinstance(build, dict):
        raise TypeError(f"Failed to get the latest build of the Actor {actor_id}.")

    data = build.get("data")
    if data is None:
        raise ValueError(f"Failed to get the latest build data of the Actor {actor_id}.")

    return data


def prune_actor_input_schema(input_schema: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Get the input schema from the Actor build and trim descriptions.

    Args:
        input_schema: The input schema dict from the Actor build.

    Returns:
        A tuple of (pruned properties dict, required field names list).
    """
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    properties_out: Dict[str, Any] = {}
    for item, meta in properties.items():
        properties_out[item] = {}
        if desc := meta.get("description"):
            properties_out[item]["description"] = (
                desc[:MAX_DESCRIPTION_LEN] + "..." if len(desc) > MAX_DESCRIPTION_LEN else desc
            )
        for key_name in ("type", "default", "prefill", "enum"):
            if value := meta.get(key_name):
                properties_out[item][key_name] = value

    return properties_out, required
