"""
Apify Actor Toolkit for Upsonic Framework.

This module provides Apify platform integration, allowing you to run any Apify Actor
as a tool. Actors are dynamically registered based on their input schemas fetched
from the Apify API.

Required Environment Variables:
-----------------------------
- APIFY_API_TOKEN: Apify API token from https://console.apify.com

Example Usage:
    ```python
    from upsonic import Agent, Task
    from upsonic.tools.custom_tools.apify import ApifyTools

    agent = Agent(
        "My Agent",
        tools=[
            ApifyTools(
                actors=["apify/rag-web-browser"],
                apify_api_token="your_apify_api_key",
            )
        ],
    )
    task = Task("What info can you find on https://example.com?", agent=agent)
    agent.print_do(task)
    ```
"""

import inspect
import json
import types
from os import getenv
from typing import Any, Dict, List, Optional, Union

from upsonic.tools.base import ToolKit
from upsonic.tools.config import ToolConfig
from upsonic.utils.integrations.apify import (
    MAX_DESCRIPTION_LEN,
    actor_id_to_tool_name,
    create_apify_client,
    get_actor_latest_build,
    props_to_json_schema,
    prune_actor_input_schema,
)
from upsonic.utils.printing import error_log

try:
    from apify_client import ApifyClient
    _APIFY_AVAILABLE = True
except ImportError:
    ApifyClient = None
    _APIFY_AVAILABLE = False


_SCHEMA_TYPE_MAP: Dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _make_actor_function(
    actor_id: str,
    client,
    properties: Dict[str, Any],
    required: List[str],
):
    """Create a tool function for an Apify Actor with a proper typed signature.

    Builds a real Python function whose ``inspect.signature`` contains one
    ``Parameter`` per schema property (with correct type annotations and
    defaults).  This lets ``function_schema`` generate the right JSON schema
    so the LLM knows which arguments to pass.
    """

    # -- 1.  Build inspect.Parameter list --------------------------------
    # Note: `self` is NOT included here. The function uses `self` as a
    # positional arg in the body, but `types.MethodType` binds it.
    # `inspect.signature` on a bound method automatically strips `self`.
    # However the wrapper created by `_make_tool_wrapper` uses
    # `functools.wraps` which copies `__signature__` — so we must NOT
    # include `self` or the schema generator will complain about missing
    # type annotations.
    params: List[inspect.Parameter] = []
    for name, meta in properties.items():
        annotation = _SCHEMA_TYPE_MAP.get(meta.get("type", ""), Any)
        if name in required:
            default = inspect.Parameter.empty
        else:
            default = meta.get("default", meta.get("prefill", None))
        params.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )

    sig = inspect.Signature(params, return_annotation=str)

    # -- 2.  The actual implementation -----------------------------------
    def actor_function(self, **kwargs: Any) -> str:
        """Run an Apify Actor."""
        try:
            details = client.actor(actor_id=actor_id).call(run_input=kwargs)
            if details is None:
                raise ValueError(
                    f"Actor: {actor_id} was not started properly and "
                    "details about the run were not returned"
                )

            run_id = details.get("id")
            if run_id is None:
                raise ValueError(f"Run ID not found in the run details for Actor: {actor_id}")

            run = client.run(run_id=run_id)
            results = run.dataset().list_items(clean=True).items

            return json.dumps(results)
        except Exception as e:
            error_log(f"Error running Apify Actor {actor_id}: {e}")
            return json.dumps([{"error": f"Error running Apify Actor {actor_id}: {e}"}])

    # -- 3.  Patch the signature so schema generation works --------------
    actor_function.__signature__ = sig
    # Also set __annotations__ for _typing_extra.get_function_type_hints
    annotations: Dict[str, type] = {"return": str}
    for p in params:
        if p.name != "self" and p.annotation is not inspect.Parameter.empty:
            annotations[p.name] = p.annotation
    actor_function.__annotations__ = annotations

    return actor_function


class ApifyTools(ToolKit):
    """Apify Actor toolkit. Dynamically registers Apify Actors as tools."""

    def __init__(
        self,
        actors: Optional[Union[str, List[str]]] = None,
        apify_api_token: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ApifyTools toolkit.

        Args:
            actors: Single Actor ID or list of Actor IDs to register as tools.
            apify_api_token: Apify API token. Falls back to APIFY_API_TOKEN env var.
            **kwargs: ToolKit params (include_tools, exclude_tools, timeout, etc.).
        """
        super().__init__(**kwargs)

        if not _APIFY_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="apify-client",
                install_command="pip install apify-client",
                feature_name="Apify tools",
            )

        self.apify_api_token: str = apify_api_token or getenv("APIFY_API_TOKEN", "")
        if not self.apify_api_token:
            raise ValueError(
                "Apify API token is required. Set APIFY_API_TOKEN environment "
                "variable or pass apify_api_token parameter."
            )

        self.client = create_apify_client(self.apify_api_token)

        if actors:
            actor_list = [actors] if isinstance(actors, str) else actors
            for actor_id in actor_list:
                self._register_actor(actor_id)

    def _register_actor(self, actor_id: str) -> None:
        """Register an Apify Actor as a tool method on this toolkit.

        Args:
            actor_id: ID of the Apify Actor (e.g. 'apify/web-scraper').
        """
        try:
            build = get_actor_latest_build(self.client, actor_id)
            tool_name = actor_id_to_tool_name(actor_id)

            actor_description = build.get("actorDefinition", {}).get("description", "")
            if len(actor_description) > MAX_DESCRIPTION_LEN:
                actor_description = actor_description[:MAX_DESCRIPTION_LEN] + "...(TRUNCATED)"

            actor_input = build.get("actorDefinition", {}).get("input")
            if not actor_input:
                raise ValueError(f"Input schema not found in the Actor build for Actor: {actor_id}")

            properties, required = prune_actor_input_schema(actor_input)

            # Build docstring with parameter descriptions from Actor's input schema
            docstring = f"{actor_description}\n\nArgs:\n"
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "No description available")
                required_marker = "(required)" if param_name in required else "(optional)"
                docstring += f"    {param_name} ({param_type}): {required_marker} {param_desc}\n"
            docstring += "\nReturns:\n    str: JSON string containing the Actor's output dataset\n"

            # Create the function and set metadata
            func = _make_actor_function(actor_id, self.client, properties, required)
            func.__name__ = tool_name
            func.__qualname__ = f"ApifyTools.{tool_name}"
            func.__doc__ = docstring

            # Mark as tool (same attributes the @tool decorator sets)
            func._upsonic_tool_config = ToolConfig()
            func._upsonic_is_tool = True
            # Provide a rich JSON schema override so the LLM gets accurate
            # array item types, enums, nested objects, and Apify editor types.
            func._json_schema_override = props_to_json_schema(properties, required)

            # Bind as a method on this instance so inspect.ismethod picks it up
            bound_method = types.MethodType(func, self)
            setattr(self, tool_name, bound_method)

        except Exception as e:
            error_log(f"Failed to register Apify Actor '{actor_id}': {e}")
