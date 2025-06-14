from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.mcp import MCPServerStdio
from ..error_wrapper import upsonic_error_handler


@upsonic_error_handler(max_retries=2, show_error_details=True)
async def agent_create(agent_model, single_task):

    mcp_servers = []

    if len(single_task.tools) > 0:
        # For loop through the tools
        for tool in single_task.tools:


            if isinstance(tool, type):
       
                # Some times the env is not dict at that situations we need to handle that
                if hasattr(tool, 'env') and isinstance(tool.env, dict):
                    env = tool.env
                else:
                    env = {}

                command = getattr(tool, 'command', None)
                args = getattr(tool, 'args', [])


                the_mcp_server = MCPServerStdio(
                    command,
                    args=args,
                    env=env,
                )

                mcp_servers.append(the_mcp_server)

                


    the_agent = PydanticAgent(agent_model, output_type=single_task.response_format, system_prompt="", end_strategy="exhaustive", retries=5, mcp_servers=mcp_servers)

    return the_agent