"""
Comprehensive HITL Test - Full Attribute and Content Verification

Tests External Tool, Cancel Run, and Durable Execution with comprehensive
verification of all attributes in:
- AgentSession
- AgentRunOutput
- RunRequirement
- PipelineExecutionStats
- ToolExecution
- StepResult
- ModelResponse, RequestUsage, ModelProfile, AgentRunInput, ModelMessage
"""

import asyncio
import os
import time
from typing import Any, Dict, List
from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.run.base import RunStatus
from upsonic.run.cancel import cancel_run
from upsonic.db.database import SqliteDatabase
from upsonic.session.agent import AgentSession
from upsonic.agent.pipeline.step import inject_error_into_step, clear_error_injection, StepResult, StepStatus
from upsonic.run.agent.output import AgentRunOutput
from upsonic.run.requirements import RunRequirement
from upsonic.run.tools.tools import ToolExecution
from upsonic.run.pipeline.stats import PipelineExecutionStats
from upsonic.run.agent.input import AgentRunInput
from upsonic.usage import RequestUsage
from upsonic.profiles import ModelProfile
from upsonic.messages.messages import ModelResponse, ModelRequest

DEBUG = True
DB_FILE = "test_comprehensive.db"


def cleanup():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    clear_error_injection()


@tool(external_execution=True)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email - requires external execution."""
    pass


@tool
def long_running_task(seconds: int) -> str:
    """A task that takes time to complete."""
    time.sleep(seconds)
    return f"Completed after {seconds} seconds"


@tool
def simple_math(a: int, b: int) -> int:
    """Perform simple math addition."""
    return a + b


def execute_tool_externally(requirement: RunRequirement) -> str:
    """Execute external tool and return result."""
    tool_exec = requirement.tool_execution
    tool_name = tool_exec.tool_name
    tool_args = tool_exec.tool_args
    
    if tool_name == "send_email":
        return f"Email sent successfully to {tool_args.get('to', 'unknown')}"
    return f"Executed external tool: {tool_name}"


# =============================================================================
# DEEP VERIFICATION FUNCTIONS
# =============================================================================

def verify_tool_execution(te: ToolExecution, description: str, 
                          expected_tool_name: str = None,
                          expected_tool_args: Dict[str, Any] = None,
                          expected_result: str = None,
                          expected_external: bool = None) -> ToolExecution:
    """Comprehensive ToolExecution verification."""
    print(f"\n  === ToolExecution: {description} ===")
    
    assert te is not None, f"FAIL [{description}]: ToolExecution is None"
    
    # Verify tool_call_id
    assert te.tool_call_id is not None, f"FAIL [{description}]: tool_call_id is None"
    assert isinstance(te.tool_call_id, str), f"FAIL [{description}]: tool_call_id should be str"
    assert len(te.tool_call_id) > 0, f"FAIL [{description}]: tool_call_id should not be empty"
    
    # Verify tool_name
    assert te.tool_name is not None, f"FAIL [{description}]: tool_name is None"
    assert isinstance(te.tool_name, str), f"FAIL [{description}]: tool_name should be str"
    if expected_tool_name:
        assert te.tool_name == expected_tool_name, f"FAIL [{description}]: tool_name should be '{expected_tool_name}', got '{te.tool_name}'"
    
    # Verify tool_args
    assert te.tool_args is not None, f"FAIL [{description}]: tool_args is None"
    assert isinstance(te.tool_args, dict), f"FAIL [{description}]: tool_args should be dict"
    if expected_tool_args:
        for key, value in expected_tool_args.items():
            assert key in te.tool_args, f"FAIL [{description}]: tool_args missing key '{key}'"
            assert te.tool_args[key] == value, f"FAIL [{description}]: tool_args['{key}'] = '{te.tool_args[key]}', expected '{value}'"
    
    # Verify created_at
    assert te.created_at is not None, f"FAIL [{description}]: created_at is None"
    assert isinstance(te.created_at, int), f"FAIL [{description}]: created_at should be int"
    assert te.created_at > 0, f"FAIL [{description}]: created_at should be > 0"
    
    # Verify result if expected
    if expected_result is not None:
        assert te.result == expected_result, f"FAIL [{description}]: result = '{te.result}', expected '{expected_result}'"
    
    # Verify external_execution_required if expected
    if expected_external is not None:
        assert te.external_execution_required == expected_external, f"FAIL [{description}]: external_execution_required = {te.external_execution_required}, expected {expected_external}"
    
    print(f"    tool_call_id: {te.tool_call_id}")
    print(f"    tool_name: {te.tool_name}")
    print(f"    tool_args: {te.tool_args}")
    print(f"    result: {te.result}")
    print(f"    external_execution_required: {te.external_execution_required}")
    print(f"    created_at: {te.created_at}")
    
    return te


def verify_step_result(sr: StepResult, description: str,
                       expected_name: str = None,
                       expected_status: StepStatus = None) -> StepResult:
    """Comprehensive StepResult verification."""
    print(f"\n  === StepResult: {description} ===")
    
    assert sr is not None, f"FAIL [{description}]: StepResult is None"
    
    # Verify name
    assert sr.name is not None, f"FAIL [{description}]: name is None"
    assert isinstance(sr.name, str), f"FAIL [{description}]: name should be str"
    if expected_name:
        assert sr.name == expected_name, f"FAIL [{description}]: name = '{sr.name}', expected '{expected_name}'"
    
    # Verify step_number
    assert sr.step_number is not None, f"FAIL [{description}]: step_number is None"
    assert isinstance(sr.step_number, int), f"FAIL [{description}]: step_number should be int"
    assert sr.step_number >= 0, f"FAIL [{description}]: step_number should be >= 0"
    
    # Verify status
    assert sr.status is not None, f"FAIL [{description}]: status is None"
    assert isinstance(sr.status, StepStatus), f"FAIL [{description}]: status should be StepStatus"
    if expected_status:
        assert sr.status == expected_status, f"FAIL [{description}]: status = {sr.status}, expected {expected_status}"
    
    # Verify execution_time
    assert sr.execution_time is not None, f"FAIL [{description}]: execution_time is None"
    assert isinstance(sr.execution_time, (int, float)), f"FAIL [{description}]: execution_time should be numeric"
    assert sr.execution_time >= 0, f"FAIL [{description}]: execution_time should be >= 0"
    
    print(f"    name: {sr.name}")
    print(f"    step_number: {sr.step_number}")
    print(f"    status: {sr.status}")
    print(f"    execution_time: {sr.execution_time:.4f}s")
    print(f"    message: {sr.message}")
    
    return sr


def verify_pipeline_stats(stats: PipelineExecutionStats, description: str,
                          expected_resumed_from: int = None) -> PipelineExecutionStats:
    """Comprehensive PipelineExecutionStats verification."""
    print(f"\n  === PipelineExecutionStats: {description} ===")
    
    assert stats is not None, f"FAIL [{description}]: PipelineExecutionStats is None"
    
    # Verify total_steps
    assert stats.total_steps is not None, f"FAIL [{description}]: total_steps is None"
    assert isinstance(stats.total_steps, int), f"FAIL [{description}]: total_steps should be int"
    assert stats.total_steps > 0, f"FAIL [{description}]: total_steps should be > 0"
    
    # Verify executed_steps
    assert stats.executed_steps is not None, f"FAIL [{description}]: executed_steps is None"
    assert isinstance(stats.executed_steps, int), f"FAIL [{description}]: executed_steps should be int"
    assert stats.executed_steps >= 0, f"FAIL [{description}]: executed_steps should be >= 0"
    
    # Verify step_timing
    assert stats.step_timing is not None, f"FAIL [{description}]: step_timing is None"
    assert isinstance(stats.step_timing, dict), f"FAIL [{description}]: step_timing should be dict"
    
    # Verify step_statuses
    assert stats.step_statuses is not None, f"FAIL [{description}]: step_statuses is None"
    assert isinstance(stats.step_statuses, dict), f"FAIL [{description}]: step_statuses should be dict"
    
    # Verify resumed_from if expected
    if expected_resumed_from is not None:
        assert stats.resumed_from == expected_resumed_from, f"FAIL [{description}]: resumed_from = {stats.resumed_from}, expected {expected_resumed_from}"
    
    print(f"    total_steps: {stats.total_steps}")
    print(f"    executed_steps: {stats.executed_steps}")
    print(f"    resumed_from: {stats.resumed_from}")
    print(f"    step_timing keys: {list(stats.step_timing.keys())[:5]}...")
    print(f"    step_statuses keys: {list(stats.step_statuses.keys())[:5]}...")
    
    return stats


def verify_run_requirement(req: RunRequirement, description: str,
                           expected_resolved: bool = None,
                           expected_tool_name: str = None,
                           expected_tool_args: Dict[str, Any] = None,
                           expected_result: str = None) -> RunRequirement:
    """Comprehensive RunRequirement verification."""
    print(f"\n  === RunRequirement: {description} ===")
    
    assert req is not None, f"FAIL [{description}]: RunRequirement is None"
    
    # Verify id
    assert req.id is not None, f"FAIL [{description}]: id is None"
    assert isinstance(req.id, str), f"FAIL [{description}]: id should be str"
    assert len(req.id) > 0, f"FAIL [{description}]: id should not be empty"
    
    # Verify created_at
    assert req.created_at is not None, f"FAIL [{description}]: created_at is None"
    
    # Verify resolved state if expected
    if expected_resolved is not None:
        actual_resolved = req.is_resolved
        assert actual_resolved == expected_resolved, f"FAIL [{description}]: is_resolved = {actual_resolved}, expected {expected_resolved}"
    
    # Verify tool_execution if expected
    if expected_tool_name:
        assert req.tool_execution is not None, f"FAIL [{description}]: tool_execution is None"
        verify_tool_execution(
            req.tool_execution, 
            f"{description} - tool_execution",
            expected_tool_name=expected_tool_name,
            expected_tool_args=expected_tool_args,
            expected_result=expected_result
        )
    
    print(f"    id: {req.id}")
    print(f"    is_resolved: {req.is_resolved}")
    print(f"    needs_external_execution: {req.needs_external_execution}")
    print(f"    has_result: {req.has_result}")
    
    return req


def verify_messages_list(messages: List, description: str, min_count: int = 0) -> List:
    """Comprehensive messages list verification."""
    print(f"\n  === Messages List: {description} ===")
    
    assert messages is not None, f"FAIL [{description}]: messages is None"
    assert isinstance(messages, list), f"FAIL [{description}]: messages should be list"
    assert len(messages) >= min_count, f"FAIL [{description}]: messages count = {len(messages)}, expected >= {min_count}"
    
    print(f"    Total messages: {len(messages)}")
    
    for i, msg in enumerate(messages[:3]):
        verify_model_message(msg, description, i)
    
    if len(messages) > 3:
        print(f"    ... and {len(messages) - 3} more messages")
    
    return messages


def verify_request_usage(usage: Any, description: str) -> RequestUsage:
    """Comprehensive RequestUsage verification with all attributes."""
    print(f"\n  === RequestUsage: {description} ===")
    
    if usage is None:
        print("    (No usage data - skipping)")
        return None
    
    # Verify it's the correct type
    assert isinstance(usage, RequestUsage), f"FAIL [{description}]: Expected RequestUsage, got {type(usage).__name__}"
    
    # === Token counts ===
    assert hasattr(usage, 'input_tokens'), f"FAIL [{description}]: Missing input_tokens"
    assert isinstance(usage.input_tokens, int), f"FAIL [{description}]: input_tokens should be int"
    assert usage.input_tokens >= 0, f"FAIL [{description}]: input_tokens should be >= 0"
    print(f"    input_tokens: {usage.input_tokens}")
    
    assert hasattr(usage, 'output_tokens'), f"FAIL [{description}]: Missing output_tokens"
    assert isinstance(usage.output_tokens, int), f"FAIL [{description}]: output_tokens should be int"
    assert usage.output_tokens >= 0, f"FAIL [{description}]: output_tokens should be >= 0"
    print(f"    output_tokens: {usage.output_tokens}")
    
    # === Calculated properties ===
    assert hasattr(usage, 'total_tokens'), f"FAIL [{description}]: Missing total_tokens"
    expected_total = usage.input_tokens + usage.output_tokens
    assert usage.total_tokens == expected_total, f"FAIL [{description}]: total_tokens mismatch: {usage.total_tokens} != {expected_total}"
    print(f"    total_tokens: {usage.total_tokens}")
    
    # === Cache tokens ===
    assert hasattr(usage, 'cache_write_tokens'), f"FAIL [{description}]: Missing cache_write_tokens"
    assert isinstance(usage.cache_write_tokens, int), f"FAIL [{description}]: cache_write_tokens should be int"
    print(f"    cache_write_tokens: {usage.cache_write_tokens}")
    
    assert hasattr(usage, 'cache_read_tokens'), f"FAIL [{description}]: Missing cache_read_tokens"
    assert isinstance(usage.cache_read_tokens, int), f"FAIL [{description}]: cache_read_tokens should be int"
    print(f"    cache_read_tokens: {usage.cache_read_tokens}")
    
    # === Audio tokens ===
    assert hasattr(usage, 'input_audio_tokens'), f"FAIL [{description}]: Missing input_audio_tokens"
    assert hasattr(usage, 'output_audio_tokens'), f"FAIL [{description}]: Missing output_audio_tokens"
    print(f"    input_audio_tokens: {usage.input_audio_tokens}")
    print(f"    output_audio_tokens: {usage.output_audio_tokens}")
    
    # === Details dict ===
    assert hasattr(usage, 'details'), f"FAIL [{description}]: Missing details"
    assert isinstance(usage.details, dict), f"FAIL [{description}]: details should be dict"
    print(f"    details: {usage.details}")
    
    return usage


def verify_model_response(response: Any, description: str, 
                          verify_parts: bool = True,
                          min_parts: int = 0) -> ModelResponse:
    """Comprehensive ModelResponse verification with all attributes."""
    print(f"\n  === ModelResponse: {description} ===")
    
    if response is None:
        print("    (No response - skipping)")
        return None
    
    # Verify type
    assert isinstance(response, ModelResponse), f"FAIL [{description}]: Expected ModelResponse, got {type(response).__name__}"
    
    # === Parts ===
    assert hasattr(response, 'parts'), f"FAIL [{description}]: Missing parts"
    assert response.parts is not None, f"FAIL [{description}]: parts is None"
    assert len(response.parts) >= min_parts, f"FAIL [{description}]: Expected >= {min_parts} parts, got {len(response.parts)}"
    print(f"    parts_count: {len(response.parts)}")
    
    if verify_parts and response.parts:
        for i, part in enumerate(response.parts[:3]):
            part_type = type(part).__name__
            print(f"      part[{i}]: {part_type}")
            # Verify each part has expected attributes
            assert hasattr(part, 'part_kind'), f"FAIL [{description}]: part[{i}] missing part_kind"
            if hasattr(part, 'content'):
                content_preview = str(part.content)[:40] + "..." if len(str(part.content)) > 40 else str(part.content)
                print(f"        content: {content_preview}")
    
    # === Usage ===
    assert hasattr(response, 'usage'), f"FAIL [{description}]: Missing usage"
    if response.usage:
        verify_request_usage(response.usage, f"{description} - response.usage")
    
    # === Model name ===
    assert hasattr(response, 'model_name'), f"FAIL [{description}]: Missing model_name"
    print(f"    model_name: {response.model_name}")
    
    # === Provider info ===
    assert hasattr(response, 'provider_name'), f"FAIL [{description}]: Missing provider_name"
    print(f"    provider_name: {response.provider_name}")
    
    assert hasattr(response, 'provider_response_id'), f"FAIL [{description}]: Missing provider_response_id"
    print(f"    provider_response_id: {response.provider_response_id}")
    
    # === Timestamp ===
    assert hasattr(response, 'timestamp'), f"FAIL [{description}]: Missing timestamp"
    assert response.timestamp is not None, f"FAIL [{description}]: timestamp is None"
    print(f"    timestamp: {response.timestamp}")
    
    # === Kind ===
    assert hasattr(response, 'kind'), f"FAIL [{description}]: Missing kind"
    assert response.kind == 'response', f"FAIL [{description}]: kind should be 'response', got '{response.kind}'"
    print(f"    kind: {response.kind}")
    
    # === Finish reason ===
    assert hasattr(response, 'finish_reason'), f"FAIL [{description}]: Missing finish_reason"
    print(f"    finish_reason: {response.finish_reason}")
    
    # === Text property ===
    assert hasattr(response, 'text'), f"FAIL [{description}]: Missing text property"
    if response.text:
        text_preview = response.text[:50] + "..." if len(response.text) > 50 else response.text
        print(f"    text: {text_preview}")
    
    # === Tool calls property ===
    assert hasattr(response, 'tool_calls'), f"FAIL [{description}]: Missing tool_calls property"
    print(f"    tool_calls count: {len(response.tool_calls)}")
    
    return response


def verify_model_profile(profile: Any, description: str) -> ModelProfile:
    """Comprehensive ModelProfile verification with all attributes."""
    print(f"\n  === ModelProfile: {description} ===")
    
    if profile is None:
        print("    (No profile - skipping)")
        return None
    
    # Verify type
    assert isinstance(profile, ModelProfile), f"FAIL [{description}]: Expected ModelProfile, got {type(profile).__name__}"
    
    # === Core capabilities ===
    assert hasattr(profile, 'supports_tools'), f"FAIL [{description}]: Missing supports_tools"
    assert isinstance(profile.supports_tools, bool), f"FAIL [{description}]: supports_tools should be bool"
    print(f"    supports_tools: {profile.supports_tools}")
    
    assert hasattr(profile, 'supports_json_schema_output'), f"FAIL [{description}]: Missing supports_json_schema_output"
    assert isinstance(profile.supports_json_schema_output, bool), f"FAIL [{description}]: supports_json_schema_output should be bool"
    print(f"    supports_json_schema_output: {profile.supports_json_schema_output}")
    
    assert hasattr(profile, 'supports_json_object_output'), f"FAIL [{description}]: Missing supports_json_object_output"
    assert isinstance(profile.supports_json_object_output, bool), f"FAIL [{description}]: supports_json_object_output should be bool"
    print(f"    supports_json_object_output: {profile.supports_json_object_output}")
    
    assert hasattr(profile, 'supports_image_output'), f"FAIL [{description}]: Missing supports_image_output"
    assert isinstance(profile.supports_image_output, bool), f"FAIL [{description}]: supports_image_output should be bool"
    print(f"    supports_image_output: {profile.supports_image_output}")
    
    # === Structured output mode ===
    assert hasattr(profile, 'default_structured_output_mode'), f"FAIL [{description}]: Missing default_structured_output_mode"
    print(f"    default_structured_output_mode: {profile.default_structured_output_mode}")
    
    # === Thinking tags ===
    assert hasattr(profile, 'thinking_tags'), f"FAIL [{description}]: Missing thinking_tags"
    assert isinstance(profile.thinking_tags, tuple), f"FAIL [{description}]: thinking_tags should be tuple"
    assert len(profile.thinking_tags) == 2, f"FAIL [{description}]: thinking_tags should have 2 elements"
    print(f"    thinking_tags: {profile.thinking_tags}")
    
    # === Prompt template ===
    assert hasattr(profile, 'prompted_output_template'), f"FAIL [{description}]: Missing prompted_output_template"
    assert isinstance(profile.prompted_output_template, str), f"FAIL [{description}]: prompted_output_template should be str"
    print(f"    prompted_output_template: (len={len(profile.prompted_output_template)})")
    
    return profile


def verify_agent_run_input(input_obj: Any, description: str) -> AgentRunInput:
    """Comprehensive AgentRunInput verification with all attributes."""
    print(f"\n  === AgentRunInput: {description} ===")
    
    if input_obj is None:
        print("    (No input - skipping)")
        return None
    
    # Verify type
    assert isinstance(input_obj, AgentRunInput), f"FAIL [{description}]: Expected AgentRunInput, got {type(input_obj).__name__}"
    
    # === User prompt ===
    assert hasattr(input_obj, 'user_prompt'), f"FAIL [{description}]: Missing user_prompt"
    assert input_obj.user_prompt is not None, f"FAIL [{description}]: user_prompt is None"
    prompt_preview = str(input_obj.user_prompt)[:60] + "..." if len(str(input_obj.user_prompt)) > 60 else str(input_obj.user_prompt)
    print(f"    user_prompt: {prompt_preview}")
    
    # === Images ===
    assert hasattr(input_obj, 'images'), f"FAIL [{description}]: Missing images"
    if input_obj.images:
        assert isinstance(input_obj.images, list), f"FAIL [{description}]: images should be list"
        print(f"    images count: {len(input_obj.images)}")
        for i, img in enumerate(input_obj.images[:2]):
            assert hasattr(img, 'data'), f"FAIL [{description}]: image[{i}] missing data"
            assert hasattr(img, 'media_type'), f"FAIL [{description}]: image[{i}] missing media_type"
            print(f"      image[{i}]: media_type={img.media_type}, data_len={len(img.data) if img.data else 0}")
    else:
        print("    images: None")
    
    # === Documents ===
    assert hasattr(input_obj, 'documents'), f"FAIL [{description}]: Missing documents"
    if input_obj.documents:
        assert isinstance(input_obj.documents, list), f"FAIL [{description}]: documents should be list"
        print(f"    documents count: {len(input_obj.documents)}")
        for i, doc in enumerate(input_obj.documents[:2]):
            assert hasattr(doc, 'data'), f"FAIL [{description}]: document[{i}] missing data"
            assert hasattr(doc, 'media_type'), f"FAIL [{description}]: document[{i}] missing media_type"
            print(f"      document[{i}]: media_type={doc.media_type}, data_len={len(doc.data) if doc.data else 0}")
    else:
        print("    documents: None")
    
    return input_obj


def verify_model_message(msg: Any, description: str, index: int) -> Any:
    """Comprehensive verification of a single ModelMessage (ModelRequest or ModelResponse)."""
    print(f"    message[{index}]: ", end="")
    
    assert msg is not None, f"FAIL [{description}]: message[{index}] is None"
    
    msg_type = type(msg).__name__
    print(f"type={msg_type}")
    
    # ModelResponse specific
    if isinstance(msg, ModelResponse):
        assert hasattr(msg, 'parts'), f"FAIL [{description}]: ModelResponse[{index}] missing parts"
        assert hasattr(msg, 'usage'), f"FAIL [{description}]: ModelResponse[{index}] missing usage"
        assert hasattr(msg, 'kind'), f"FAIL [{description}]: ModelResponse[{index}] missing kind"
        assert msg.kind == 'response', f"FAIL [{description}]: ModelResponse[{index}] kind should be 'response'"
        print(f"      parts_count: {len(msg.parts)}, kind: {msg.kind}")
        
        # Verify parts structure
        for j, part in enumerate(msg.parts[:2]):
            part_type = type(part).__name__
            assert hasattr(part, 'part_kind'), f"FAIL [{description}]: ModelResponse[{index}].part[{j}] missing part_kind"
            print(f"        part[{j}]: {part_type} (kind={part.part_kind})")
    
    # ModelRequest specific
    elif isinstance(msg, ModelRequest):
        assert hasattr(msg, 'parts'), f"FAIL [{description}]: ModelRequest[{index}] missing parts"
        assert hasattr(msg, 'kind'), f"FAIL [{description}]: ModelRequest[{index}] missing kind"
        assert msg.kind == 'request', f"FAIL [{description}]: ModelRequest[{index}] kind should be 'request'"
        print(f"      parts_count: {len(msg.parts)}, kind: {msg.kind}")
        
        # Verify parts structure  
        for j, part in enumerate(msg.parts[:2]):
            part_type = type(part).__name__
            assert hasattr(part, 'part_kind'), f"FAIL [{description}]: ModelRequest[{index}].part[{j}] missing part_kind"
            print(f"        part[{j}]: {part_type} (kind={part.part_kind})")
    
    # Generic dict (from deserialization)
    elif isinstance(msg, dict):
        print(f"      keys: {list(msg.keys())}")
        if 'parts' in msg:
            print(f"      parts_count: {len(msg['parts'])}")
    
    else:
        # Fallback for other types with parts
        if hasattr(msg, 'parts'):
            print(f"      parts_count: {len(msg.parts)}")
    
    return msg


def verify_messages_list_deep(messages: List, description: str, 
                               min_count: int = 0,
                               verify_content: bool = True) -> List:
    """Deep verification of messages list with content checks."""
    print(f"\n  === Messages List Deep Verification: {description} ===")
    
    assert messages is not None, f"FAIL [{description}]: messages is None"
    assert isinstance(messages, list), f"FAIL [{description}]: messages should be list"
    assert len(messages) >= min_count, f"FAIL [{description}]: messages count = {len(messages)}, expected >= {min_count}"
    
    print(f"    Total messages: {len(messages)}")
    
    if verify_content:
        for i, msg in enumerate(messages[:5]):
            verify_model_message(msg, description, i)
    
    if len(messages) > 5:
        print(f"    ... and {len(messages) - 5} more messages")
    
    return messages


def verify_chat_history_deep(chat_history: List, description: str,
                              min_count: int = 1) -> List:
    """Deep verification of chat_history with structure validation."""
    print(f"\n  === Chat History Deep Verification: {description} ===")
    
    assert chat_history is not None, f"FAIL [{description}]: chat_history is None"
    assert isinstance(chat_history, list), f"FAIL [{description}]: chat_history should be list"
    assert len(chat_history) >= min_count, f"FAIL [{description}]: chat_history count = {len(chat_history)}, expected >= {min_count}"
    
    print(f"    Total messages: {len(chat_history)}")
    
    # Verify structure of each message
    request_count = 0
    response_count = 0
    
    for i, msg in enumerate(chat_history):
        if isinstance(msg, ModelRequest):
            request_count += 1
        elif isinstance(msg, ModelResponse):
            response_count += 1
    
    print(f"    ModelRequest count: {request_count}")
    print(f"    ModelResponse count: {response_count}")
    
    # Verify first few messages in detail
    for i, msg in enumerate(chat_history[:3]):
        verify_model_message(msg, description, i)
    
    if len(chat_history) > 3:
        print(f"    ... and {len(chat_history) - 3} more messages")
    
    return chat_history


def compare_messages_content(msg1: Any, msg2: Any, index: int, description: str) -> bool:
    """Compare two message objects for content equality."""
    # Both should be same type
    if type(msg1).__name__ != type(msg2).__name__:
        print(f"      message[{index}]: TYPE MISMATCH - {type(msg1).__name__} vs {type(msg2).__name__}")
        return False
    
    # Both should have parts
    if hasattr(msg1, 'parts') and hasattr(msg2, 'parts'):
        if len(msg1.parts) != len(msg2.parts):
            print(f"      message[{index}]: PARTS COUNT MISMATCH - {len(msg1.parts)} vs {len(msg2.parts)}")
            return False
        
        # Compare parts
        for j, (p1, p2) in enumerate(zip(msg1.parts, msg2.parts)):
            if type(p1).__name__ != type(p2).__name__:
                print(f"        part[{j}]: TYPE MISMATCH - {type(p1).__name__} vs {type(p2).__name__}")
                return False
            
            if hasattr(p1, 'content') and hasattr(p2, 'content'):
                if str(p1.content) != str(p2.content):
                    print(f"        part[{j}]: CONTENT MISMATCH")
                    return False
    
    # For ModelResponse, check kind
    if hasattr(msg1, 'kind') and hasattr(msg2, 'kind'):
        if msg1.kind != msg2.kind:
            print(f"      message[{index}]: KIND MISMATCH - {msg1.kind} vs {msg2.kind}")
            return False
    
    return True


def compare_messages_lists(list1: List, list2: List, description: str) -> None:
    """Comprehensive comparison of two message lists."""
    print(f"\n{'='*60}")
    print(f"Messages List Comparison: {description}")
    print(f"{'='*60}")
    
    assert list1 is not None, f"FAIL [{description}]: list1 is None"
    assert list2 is not None, f"FAIL [{description}]: list2 is None"
    assert isinstance(list1, list), f"FAIL [{description}]: list1 should be list"
    assert isinstance(list2, list), f"FAIL [{description}]: list2 should be list"
    
    print(f"  list1 count: {len(list1)}")
    print(f"  list2 count: {len(list2)}")
    
    # Lengths should match
    assert len(list1) == len(list2), f"FAIL [{description}]: Length mismatch - {len(list1)} vs {len(list2)}"
    print(f"  ✓ Lengths match: {len(list1)}")
    
    # Compare each message
    all_match = True
    for i in range(min(len(list1), len(list2))):
        if not compare_messages_content(list1[i], list2[i], i, description):
            all_match = False
            print(f"  ✗ Message[{i}] content mismatch")
        else:
            print(f"  ✓ Message[{i}] matches")
    
    assert all_match, f"FAIL [{description}]: Message content mismatch detected"
    print("\n  ✓ ALL MESSAGES MATCH")


def verify_session_messages_consistency(session: AgentSession, output: AgentRunOutput, description: str) -> None:
    """Verify that AgentSession.messages and AgentRunOutput.chat_history are consistent."""
    print(f"\n{'='*60}")
    print(f"Session Messages vs Output Chat History: {description}")
    print(f"{'='*60}")
    
    # Get session messages
    session_messages = session.messages if session.messages else []
    output_chat_history = output.chat_history if output.chat_history else []
    
    print(f"  session.messages count: {len(session_messages)}")
    print(f"  output.chat_history count: {len(output_chat_history)}")
    
    # They should have the same length (or chat_history might be longer if it includes historical)
    # For consistency, we check that chat_history contains all session messages
    if len(session_messages) > 0:
        assert len(output_chat_history) >= len(session_messages), \
            f"FAIL [{description}]: chat_history ({len(output_chat_history)}) should contain at least session.messages ({len(session_messages)})"
        
        # Compare the last N messages where N = len(session_messages)
        # This handles the case where chat_history includes historical messages
        if len(output_chat_history) >= len(session_messages):
            recent_chat_history = output_chat_history[-len(session_messages):]
            print(f"  Comparing last {len(session_messages)} messages from chat_history with session.messages")
            
            for i in range(len(session_messages)):
                if not compare_messages_content(session_messages[i], recent_chat_history[i], i, description):
                    print(f"  ✗ Message[{i}] mismatch between session.messages and chat_history")
                    assert False, f"FAIL [{description}]: Message[{i}] mismatch"
                else:
                    print(f"  ✓ Message[{i}] matches")
            
            print("\n  ✓ ALL MESSAGES CONSISTENT")
    else:
        print("  (No session messages to compare)")


def verify_storage_consistency(before_session: AgentSession, after_session: AgentSession,
                               before_output: AgentRunOutput, after_output: AgentRunOutput,
                               description: str) -> None:
    """Verify that messages and chat_history are preserved correctly through storage.
    
    Validates:
    1. Session messages are preserved (before messages are subset of after)
    2. Chat history is preserved (before history is subset of after)
    3. Session messages match output chat_history (after storage)
    """
    print(f"\n{'='*60}")
    print(f"Storage Consistency Verification: {description}")
    print(f"{'='*60}")
    
    # === 1. Verify AgentSession.messages preserved ===
    print("\n--- AgentSession.messages: Before vs After Storage ---")
    before_messages = before_session.messages if before_session.messages else []
    after_messages = after_session.messages if after_session.messages else []
    
    print(f"  Before storage: {len(before_messages)} messages")
    print(f"  After storage: {len(after_messages)} messages")
    
    # Messages should be preserved or grown (not shrunk)
    assert len(after_messages) >= len(before_messages), \
        f"FAIL [{description}]: Session messages shrunk from {len(before_messages)} to {len(after_messages)}"
    
    # Verify that before messages are preserved in after (prefix match)
    if len(before_messages) > 0:
        print(f"  Verifying first {len(before_messages)} messages are preserved...")
        for i in range(len(before_messages)):
            if not compare_messages_content(before_messages[i], after_messages[i], i, description):
                assert False, f"FAIL [{description}]: Message[{i}] not preserved in session.messages"
        print(f"  ✓ All {len(before_messages)} before messages preserved")
    print(f"  ✓ Messages preserved/grown: {len(before_messages)} -> {len(after_messages)}")
    
    # === 2. Verify AgentRunOutput.chat_history preserved ===
    print("\n--- AgentRunOutput.chat_history: Before vs After Storage ---")
    before_history = before_output.chat_history if before_output.chat_history else []
    after_history = after_output.chat_history if after_output.chat_history else []
    
    print(f"  Before storage: {len(before_history)} messages")
    print(f"  After storage: {len(after_history)} messages")
    
    # Chat history should be preserved or grown (not shrunk)
    assert len(after_history) >= len(before_history), \
        f"FAIL [{description}]: Chat history shrunk from {len(before_history)} to {len(after_history)}"
    
    # Verify that before history is preserved in after (prefix match)
    if len(before_history) > 0:
        print(f"  Verifying first {len(before_history)} messages are preserved...")
        for i in range(len(before_history)):
            if not compare_messages_content(before_history[i], after_history[i], i, description):
                assert False, f"FAIL [{description}]: Message[{i}] not preserved in chat_history"
        print(f"  ✓ All {len(before_history)} before messages preserved")
    print(f"  ✓ Chat history preserved/grown: {len(before_history)} -> {len(after_history)}")
    
    # === 3. Verify AgentSession.messages matches AgentRunOutput.chat_history (after storage) ===
    print("\n--- Consistency: Session.messages vs Output.chat_history (After Storage) ---")
    compare_messages_lists(after_messages, after_history, f"{description} - Session.messages vs chat_history")
    
    print("\n  ✓ STORAGE CONSISTENCY VERIFIED")


def verify_agent_run_output(output: AgentRunOutput, description: str,
                            expected_status: RunStatus = None,
                            expected_run_id: str = None,
                            min_messages: int = 0,
                            min_step_results: int = 0,
                            min_requirements: int = 0,
                            verify_content: bool = True) -> AgentRunOutput:
    """Comprehensive AgentRunOutput verification with all attributes."""
    print(f"\n{'='*60}")
    print(f"AgentRunOutput Verification: {description}")
    print(f"{'='*60}")
    
    assert output is not None, f"FAIL [{description}]: AgentRunOutput is None"
    
    # === Identity ===
    print("\n--- Identity ---")
    assert output.run_id is not None, f"FAIL [{description}]: run_id is None"
    assert isinstance(output.run_id, str), f"FAIL [{description}]: run_id should be str"
    if expected_run_id:
        assert output.run_id == expected_run_id, f"FAIL [{description}]: run_id mismatch"
    print(f"  run_id: {output.run_id}")
    
    assert output.agent_id is not None, f"FAIL [{description}]: agent_id is None"
    print(f"  agent_id: {output.agent_id}")
    print(f"  agent_name: {output.agent_name}")
    print(f"  session_id: {output.session_id}")
    print(f"  user_id: {output.user_id}")
    
    # === Status ===
    print("\n--- Status ---")
    assert output.status is not None, f"FAIL [{description}]: status is None"
    if expected_status:
        assert output.status == expected_status, f"FAIL [{description}]: status = {output.status}, expected {expected_status}"
    print(f"  status: {output.status}")
    print(f"  is_complete: {output.is_complete}")
    print(f"  is_paused: {output.is_paused}")
    print(f"  is_cancelled: {output.is_cancelled}")
    print(f"  is_error: {output.is_error}")
    print(f"  pause_reason: {output.pause_reason}")
    print(f"  error_details: {output.error_details}")
    
    # === Output ===
    print("\n--- Output ---")
    if output.output is not None:
        output_preview = str(output.output)[:60] + "..." if len(str(output.output)) > 60 else str(output.output)
        print(f"  output: {output_preview}")
    else:
        print("  output: None")
    print(f"  output_schema: {output.output_schema}")
    
    # === Messages (deep verification) ===
    print("\n--- Messages ---")
    if output.messages:
        verify_messages_list_deep(output.messages, f"{description} - messages", 
                                   min_count=min_messages, verify_content=True)
    else:
        print("  messages: None or empty")
    
    # === Chat History (deep verification) ===
    print("\n--- Chat History ---")
    if output.chat_history:
        verify_chat_history_deep(output.chat_history, f"{description} - chat_history", min_count=1)
    else:
        print("  chat_history: None or empty")
    
    # === Response (deep verification) ===
    verify_model_response(output.response, f"{description} - response", verify_parts=True)
    
    # === Usage (deep verification) ===
    verify_request_usage(output.usage, f"{description} - usage")
    
    # === Model Info ===
    print("\n--- Model Info ---")
    assert output.model_name is not None, f"FAIL [{description}]: model_name is None"
    assert isinstance(output.model_name, str), f"FAIL [{description}]: model_name should be str"
    print(f"  model_name: {output.model_name}")
    
    print(f"  model_provider: {output.model_provider}")
    
    # === Model Profile (deep verification) ===
    if output.model_provider_profile:
        verify_model_profile(output.model_provider_profile, f"{description} - profile")
    
    # === Tools ===
    print("\n--- Tools ---")
    print(f"  tool_call_count: {output.tool_call_count}")
    print(f"  tool_limit_reached: {output.tool_limit_reached}")
    if output.tools:
        print(f"  tools count: {len(output.tools)}")
        for i, te in enumerate(output.tools[:2]):
            verify_tool_execution(te, f"{description} - tool[{i}]")
    
    # === Requirements ===
    print("\n--- Requirements ---")
    if output.requirements:
        assert len(output.requirements) >= min_requirements, f"FAIL [{description}]: requirements count = {len(output.requirements)}, expected >= {min_requirements}"
        print(f"  requirements count: {len(output.requirements)}")
        print(f"  active_requirements count: {len(output.active_requirements)}")
        for i, req in enumerate(output.requirements[:2]):
            verify_run_requirement(req, f"{description} - requirement[{i}]")
    else:
        print("  requirements: None or empty")
    
    # === Step Results ===
    print("\n--- Step Results ---")
    if output.step_results:
        assert len(output.step_results) >= min_step_results, f"FAIL [{description}]: step_results count = {len(output.step_results)}, expected >= {min_step_results}"
        print(f"  step_results count: {len(output.step_results)}")
        for i, sr in enumerate(output.step_results[:3]):
            verify_step_result(sr, f"{description} - step[{i}]")
        if len(output.step_results) > 3:
            print(f"    ... and {len(output.step_results) - 3} more steps")
    else:
        print("  step_results: None or empty")
    
    # === Execution Stats ===
    if output.execution_stats:
        verify_pipeline_stats(output.execution_stats, description)
    
    # === Timestamps ===
    print("\n--- Timestamps ---")
    print(f"  created_at: {output.created_at}")
    print(f"  updated_at: {output.updated_at}")
    
    # === Input (deep verification) ===
    print("\n--- Input ---")
    if output.input:
        verify_agent_run_input(output.input, f"{description} - input")
    else:
        print("  input: None")
    
    # === Task ===
    print("\n--- Task ---")
    if output.task:
        print(f"  task type: {type(output.task).__name__}")
        if hasattr(output.task, 'description'):
            print(f"  task.description: {str(output.task.description)[:50]}...")
    else:
        print("  task: None")
    
    return output


def verify_agent_session(session: AgentSession, description: str,
                         expected_run_id: str = None,
                         expected_run_count: int = None) -> AgentSession:
    """Comprehensive AgentSession verification."""
    print(f"\n{'='*60}")
    print(f"AgentSession Verification: {description}")
    print(f"{'='*60}")
    
    assert session is not None, f"FAIL [{description}]: AgentSession is None"
    
    # === Identity ===
    print("\n--- Identity ---")
    assert session.session_id is not None, f"FAIL [{description}]: session_id is None"
    assert isinstance(session.session_id, str), f"FAIL [{description}]: session_id should be str"
    print(f"  session_id: {session.session_id}")
    print(f"  agent_id: {session.agent_id}")
    print(f"  user_id: {session.user_id}")
    print(f"  workflow_id: {session.workflow_id}")
    
    # === Runs ===
    print("\n--- Runs ---")
    assert session.runs is not None, f"FAIL [{description}]: runs is None"
    assert isinstance(session.runs, dict), f"FAIL [{description}]: runs should be dict"
    print(f"  runs count: {len(session.runs)}")
    print(f"  run_ids: {list(session.runs.keys())}")
    
    if expected_run_count is not None:
        assert len(session.runs) >= expected_run_count, f"FAIL [{description}]: runs count = {len(session.runs)}, expected >= {expected_run_count}"
    
    if expected_run_id:
        assert expected_run_id in session.runs, f"FAIL [{description}]: run_id {expected_run_id} not in runs"
    
    # === Messages ===
    print("\n--- Messages ---")
    if session.messages:
        print(f"  messages count: {len(session.messages)}")
    else:
        print("  messages: None or empty")
    
    # === Metadata ===
    print("\n--- Metadata ---")
    print(f"  session_data: {session.session_data}")
    print(f"  metadata: {session.metadata}")
    print(f"  summary: {session.summary}")
    
    # === Timestamps ===
    print("\n--- Timestamps ---")
    print(f"  created_at: {session.created_at}")
    print(f"  updated_at: {session.updated_at}")
    
    return session


def compare_outputs(before: AgentRunOutput, after: AgentRunOutput, description: str) -> None:
    """Compare two AgentRunOutput instances for consistency."""
    print(f"\n{'='*60}")
    print(f"Output Comparison: {description}")
    print(f"{'='*60}")
    
    # === Identity should match ===
    assert before.run_id == after.run_id, f"FAIL: run_id mismatch: {before.run_id} vs {after.run_id}"
    print(f"  run_id: MATCH ({before.run_id})")
    
    # === Chat history should be preserved or grown ===
    before_history_len = len(before.chat_history) if before.chat_history else 0
    after_history_len = len(after.chat_history) if after.chat_history else 0
    assert after_history_len >= before_history_len, f"FAIL: chat_history shrunk: {before_history_len} -> {after_history_len}"
    print(f"  chat_history: {before_history_len} -> {after_history_len} (preserved/grown)")
    
    # === Step results should be preserved or grown ===
    before_steps = len(before.step_results) if before.step_results else 0
    after_steps = len(after.step_results) if after.step_results else 0
    assert after_steps >= before_steps, f"FAIL: step_results shrunk: {before_steps} -> {after_steps}"
    print(f"  step_results: {before_steps} -> {after_steps} (preserved/grown)")
    
    # === Requirements should be preserved ===
    before_reqs = len(before.requirements) if before.requirements else 0
    after_reqs = len(after.requirements) if after.requirements else 0
    assert after_reqs >= before_reqs, f"FAIL: requirements shrunk: {before_reqs} -> {after_reqs}"
    print(f"  requirements: {before_reqs} -> {after_reqs} (preserved/grown)")
    
    print("\n  COMPARISON PASSED")


# =============================================================================
# TEST: EXTERNAL TOOL
# =============================================================================

async def test_external_tool_comprehensive():
    """Comprehensive External Tool test with full attribute verification."""
    print("\n" + "="*80)
    print("TEST: External Tool - Full Attribute Verification")
    print("="*80)
    
    cleanup()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent1 = Agent("openai/gpt-4o-mini", name="agent1", db=db, debug=DEBUG)
    task = Task(
        description="Send an email to test@example.com with subject 'Hello' and body 'Test'.",
        tools=[send_email]
    )
    
    expected_tool_args = {'to': 'test@example.com', 'subject': 'Hello', 'body': 'Test'}
    expected_result = "Email sent successfully to test@example.com"
    
    # =========================================================================
    # STEP 1: Initial run - should pause
    # =========================================================================
    print("\n[STEP 1] Running do_async (expecting pause)...")
    output = await agent1.do_async(task, return_output=True)
    run_id = output.run_id
    
    # Verify output at pause
    verify_agent_run_output(
        output, "After Initial Pause",
        expected_status=RunStatus.paused,
        expected_run_id=run_id,
        min_step_results=5,
        min_requirements=1
    )
    
    # Verify requirement details
    assert len(output.active_requirements) == 1, "Should have 1 active requirement"
    req = output.active_requirements[0]
    verify_run_requirement(
        req, "Paused Requirement",
        expected_resolved=False,
        expected_tool_name="send_email",
        expected_tool_args=expected_tool_args,
        expected_result=None
    )
    
    print("\n[STEP 1] PASSED")
    
    # =========================================================================
    # STEP 2: Verify storage after pause
    # =========================================================================
    print("\n[STEP 2] Verifying storage after pause...")
    
    # Capture in-memory state before reading from storage
    # Note: We need to get the session from the agent's memory if available
    # For now, we'll read from storage and compare
    
    session = await db.storage.read_async("test_session", AgentSession)
    verify_agent_session(session, "After Pause", expected_run_id=run_id, expected_run_count=1)
    
    stored_output = session.runs[run_id].output
    verify_agent_run_output(
        stored_output, "Stored After Pause",
        expected_status=RunStatus.paused,
        expected_run_id=run_id
    )
    
    # Compare in-memory vs stored
    compare_outputs(output, stored_output, "In-Memory vs Stored (Pause)")
    
    # Verify storage consistency: messages and chat_history
    # Create a "before" session from in-memory output for comparison
    # Since we don't have direct access to session before storage, we'll compare
    # the stored session messages with the output's chat_history
    verify_session_messages_consistency(session, stored_output, "After Pause - Storage")
    
    print("\n[STEP 2] PASSED")
    
    # =========================================================================
    # STEP 3: Set tool result
    # =========================================================================
    print("\n[STEP 3] Setting external tool result...")
    
    # Set result on the in-memory requirement
    req.set_external_execution_result(expected_result)
    
    # Verify result was set
    assert req.tool_execution.result == expected_result, "Result should be set"
    assert req.is_resolved, "Requirement should be resolved"
    
    print(f"  Result set: {expected_result}")
    print("\n[STEP 3] PASSED")
    
    # =========================================================================
    # STEP 4: Resume with NEW agent
    # =========================================================================
    print("\n[STEP 4] Resuming with new agent...")
    
    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent2 = Agent("openai/gpt-4o-mini", name="agent2", db=db2, debug=DEBUG)
    
    result = await agent2.continue_run_async(
        run_id=run_id, 
        return_output=True,
        requirements=output.requirements  # Pass all requirements (with results set)
    )
    
    # Verify completed output
    verify_agent_run_output(
        result, "After Completion",
        expected_status=RunStatus.completed,
        expected_run_id=run_id,
        min_step_results=10
    )
    
    # Verify content exists
    assert result.output is not None, "Output content should exist"
    assert len(str(result.output)) > 0, "Output content should not be empty"
    
    print(f"  Result content: {result.output}")
    print("\n[STEP 4] PASSED")
    
    # =========================================================================
    # STEP 5: Verify final storage
    # =========================================================================
    print("\n[STEP 5] Verifying final storage...")
    
    # Capture before state (from step 2)
    before_session = session
    before_output = stored_output
    
    final_session = await db2.storage.read_async("test_session", AgentSession)
    verify_agent_session(final_session, "Final", expected_run_id=run_id)
    
    final_output = final_session.runs[run_id].output
    verify_agent_run_output(
        final_output, "Final Stored",
        expected_status=RunStatus.completed,
        expected_run_id=run_id
    )
    
    # Verify requirement is resolved in storage
    if final_output.requirements:
        for req in final_output.requirements:
            if req.tool_execution and req.tool_execution.tool_name == "send_email":
                assert req.is_resolved, "Requirement should be resolved"
                assert req.tool_execution.result == expected_result, "Result should match"
    
    # Verify storage consistency: messages and chat_history before and after
    verify_storage_consistency(
        before_session, final_session,
        before_output, final_output,
        "External Tool - Pause to Completion"
    )
    
    # Verify current state consistency
    verify_session_messages_consistency(final_session, final_output, "Final - Storage")
    
    print("\n[STEP 5] PASSED")
    
    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: External Tool - Full Attribute Verification")
    print("="*80)


# =============================================================================
# TEST: CANCEL RUN
# =============================================================================

async def test_cancel_run_comprehensive():
    """Comprehensive Cancel Run test with full attribute verification."""
    print("\n" + "="*80)
    print("TEST: Cancel Run - Full Attribute Verification")
    print("="*80)
    
    cleanup()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent1 = Agent("openai/gpt-4o-mini", name="agent1", db=db, debug=DEBUG)
    task = Task(
        description="Call long_running_task with 5 seconds.",
        tools=[long_running_task]
    )
    
    # =========================================================================
    # STEP 1: Start and cancel
    # =========================================================================
    print("\n[STEP 1] Starting run and cancelling...")
    
    async def run_task():
        return await agent1.do_async(task, return_output=True)
    
    run_future = asyncio.create_task(run_task())
    await asyncio.sleep(1.5)
    
    run_id = agent1.run_id
    assert run_id is not None, "run_id should be available"
    
    cancel_run(run_id)
    print(f"  Cancelled run: {run_id}")
    
    output = await asyncio.wait_for(run_future, timeout=10.0)
    
    # Verify cancelled output
    verify_agent_run_output(
        output, "After Cancel",
        expected_status=RunStatus.cancelled,
        expected_run_id=run_id,
        min_step_results=5
    )
    
    print("\n[STEP 1] PASSED")
    
    # =========================================================================
    # STEP 2: Verify storage after cancel
    # =========================================================================
    print("\n[STEP 2] Verifying storage after cancel...")
    
    session = await db.storage.read_async("test_session", AgentSession)
    verify_agent_session(session, "After Cancel", expected_run_id=run_id)
    
    stored_output = session.runs[run_id].output
    verify_agent_run_output(
        stored_output, "Stored After Cancel",
        expected_status=RunStatus.cancelled
    )
    
    compare_outputs(output, stored_output, "In-Memory vs Stored (Cancel)")
    
    # Verify storage consistency: messages and chat_history
    verify_session_messages_consistency(session, stored_output, "After Cancel - Storage")
    
    print("\n[STEP 2] PASSED")
    
    # =========================================================================
    # STEP 3: Resume with new agent
    # =========================================================================
    print("\n[STEP 3] Resuming with new agent...")
    
    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent2 = Agent("openai/gpt-4o-mini", name="agent2", db=db2, debug=DEBUG)
    
    result = await agent2.continue_run_async(run_id=run_id, return_output=True)
    
    verify_agent_run_output(
        result, "After Resume",
        expected_status=RunStatus.completed,
        expected_run_id=run_id,
        min_step_results=10
    )
    
    print(f"  Result: {result.output}")
    print("\n[STEP 3] PASSED")
    
    # =========================================================================
    # STEP 4: Verify final storage
    # =========================================================================
    print("\n[STEP 4] Verifying final storage...")
    
    # Capture before state (from step 2)
    before_session = session
    before_output = stored_output
    
    final_session = await db2.storage.read_async("test_session", AgentSession)
    final_output = final_session.runs[run_id].output
    
    verify_agent_run_output(
        final_output, "Final Stored",
        expected_status=RunStatus.completed
    )
    
    # Compare before/after
    compare_outputs(stored_output, final_output, "Cancel vs Completed")
    
    # Verify storage consistency: messages and chat_history before and after
    verify_storage_consistency(
        before_session, final_session,
        before_output, final_output,
        "Cancel Run - Cancel to Completion"
    )
    
    # Verify current state consistency
    verify_session_messages_consistency(final_session, final_output, "Final - Storage")
    
    print("\n[STEP 4] PASSED")
    
    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Cancel Run - Full Attribute Verification")
    print("="*80)


# =============================================================================
# TEST: DURABLE EXECUTION
# =============================================================================

async def test_durable_execution_comprehensive():
    """Comprehensive Durable Execution test with full attribute verification."""
    print("\n" + "="*80)
    print("TEST: Durable Execution - Full Attribute Verification")
    print("="*80)
    
    cleanup()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent = Agent("openai/gpt-4o-mini", db=db, debug=DEBUG, retry=1)
    task = Task(
        description="What is 5 + 3? Reply with just the number.",
        tools=[simple_math]
    )
    
    # Inject error that triggers once
    inject_error_into_step("model_execution", RuntimeError, "Simulated error", trigger_count=1)
    
    # =========================================================================
    # STEP 1: Run and catch error
    # =========================================================================
    print("\n[STEP 1] Running do_async (expecting error)...")
    
    run_id = None
    error_output = None
    try:
        _ = await agent.do_async(task, return_output=True)
        assert False, "Should have raised exception"
    except Exception as e:
        print(f"  Caught error: {type(e).__name__}: {str(e)[:60]}...")
        assert "INJECTED ERROR" in str(e), "Should be injected error"
        
        error_output = getattr(agent, '_agent_run_output', None)
        if error_output:
            run_id = error_output.run_id
    
    assert run_id is not None, "run_id should be available"
    print(f"  Run ID: {run_id}")
    
    # Verify error output
    if error_output:
        verify_agent_run_output(
            error_output, "After Error",
            expected_status=RunStatus.error,
            expected_run_id=run_id,
            min_step_results=5
        )
    
    print("\n[STEP 1] PASSED")
    
    # =========================================================================
    # STEP 2: Verify storage after error
    # =========================================================================
    print("\n[STEP 2] Verifying storage after error...")
    
    session = await db.storage.read_async("test_session", AgentSession)
    verify_agent_session(session, "After Error", expected_run_id=run_id)
    
    stored_output = session.runs[run_id].output
    verify_agent_run_output(
        stored_output, "Stored After Error",
        expected_status=RunStatus.error
    )
    
    # Verify error step
    error_steps = [sr for sr in stored_output.step_results if sr.status == StepStatus.ERROR]
    assert len(error_steps) >= 1, "Should have error step"
    print(f"  Error step: {error_steps[-1].name} at step {error_steps[-1].step_number}")
    
    # Verify storage consistency: messages and chat_history
    if error_output:
        compare_outputs(error_output, stored_output, "In-Memory vs Stored (Error)")
    verify_session_messages_consistency(session, stored_output, "After Error - Storage")
    
    print("\n[STEP 2] PASSED")
    
    # =========================================================================
    # STEP 3: Resume
    # =========================================================================
    print("\n[STEP 3] Resuming...")
    
    result = await agent.continue_run_async(run_id=run_id, return_output=True)
    
    verify_agent_run_output(
        result, "After Resume",
        expected_status=RunStatus.completed,
        expected_run_id=run_id,
        min_step_results=10
    )
    
    print(f"  Result: {result.output}")
    print("\n[STEP 3] PASSED")
    
    # =========================================================================
    # STEP 4: Verify final state and compare
    # =========================================================================
    print("\n[STEP 4] Verifying final state...")
    
    # Capture before state (from step 2)
    before_session = session
    before_output = stored_output
    
    final_session = await db.storage.read_async("test_session", AgentSession)
    final_output = final_session.runs[run_id].output
    
    verify_agent_run_output(
        final_output, "Final Stored",
        expected_status=RunStatus.completed
    )
    
    # Compare error vs completed
    compare_outputs(stored_output, final_output, "Error vs Completed")
    
    # Verify execution_stats shows resumption
    if final_output.execution_stats:
        verify_pipeline_stats(final_output.execution_stats, "Final Stats")
    
    # Verify storage consistency: messages and chat_history before and after
    verify_storage_consistency(
        before_session, final_session,
        before_output, final_output,
        "Durable Execution - Error to Completion"
    )
    
    # Verify current state consistency
    verify_session_messages_consistency(final_session, final_output, "Final - Storage")
    
    print("\n[STEP 4] PASSED")
    
    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Durable Execution - Full Attribute Verification")
    print("="*80)


# =============================================================================
# TEST: CROSS-PROCESS DURABLE EXECUTION
# =============================================================================

async def test_durable_cross_process_comprehensive():
    """Durable execution with new agent (cross-process simulation)."""
    print("\n" + "="*80)
    print("TEST: Durable Execution Cross-Process - Full Verification")
    print("="*80)
    
    cleanup()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent1 = Agent("openai/gpt-4o-mini", db=db, debug=DEBUG, retry=1)
    task = Task("What is 7 + 2? Reply with just the number.")
    
    inject_error_into_step("response_processing", ValueError, "Simulated error", trigger_count=1)
    
    # =========================================================================
    # STEP 1: Error with agent1
    # =========================================================================
    print("\n[STEP 1] Running with agent1 (expecting error)...")
    
    run_id = None
    try:
        _ = await agent1.do_async(task, return_output=True)
        assert False, "Should have raised"
    except Exception as e:
        print(f"  Error: {type(e).__name__}")
        error_output = getattr(agent1, '_agent_run_output', None)
        if error_output:
            run_id = error_output.run_id
    
    assert run_id is not None, "run_id required"
    print(f"  Run ID: {run_id}")
    print("\n[STEP 1] PASSED")
    
    # =========================================================================
    # STEP 2: Verify storage
    # =========================================================================
    print("\n[STEP 2] Verifying storage...")
    
    session = await db.storage.read_async("test_session", AgentSession)
    stored_output = session.runs[run_id].output
    
    verify_agent_run_output(stored_output, "Stored Error", expected_status=RunStatus.error)
    
    # Verify storage consistency: messages and chat_history
    if error_output:
        compare_outputs(error_output, stored_output, "In-Memory vs Stored (Error)")
    verify_session_messages_consistency(session, stored_output, "After Error - Storage")
    
    print("\n[STEP 2] PASSED")
    
    # =========================================================================
    # STEP 3: Resume with NEW agent (cross-process)
    # =========================================================================
    print("\n[STEP 3] Resuming with NEW agent (cross-process)...")
    
    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent2 = Agent("openai/gpt-4o-mini", db=db2, debug=DEBUG, retry=1)
    
    assert agent2.agent_id != agent1.agent_id, "Should be different agent"
    
    result = await agent2.continue_run_async(run_id=run_id, return_output=True)
    
    verify_agent_run_output(
        result, "Completed by agent2",
        expected_status=RunStatus.completed,
        expected_run_id=run_id
    )
    
    print(f"  Result: {result.output}")
    print("\n[STEP 3] PASSED")
    
    # =========================================================================
    # STEP 4: Verify cross-process consistency
    # =========================================================================
    print("\n[STEP 4] Verifying cross-process consistency...")
    
    # Capture before state (from step 2)
    before_session = session
    before_output = stored_output
    
    final_session = await db2.storage.read_async("test_session", AgentSession)
    final_output = final_session.runs[run_id].output
    
    compare_outputs(stored_output, final_output, "Cross-Process Error vs Completed")
    
    # Verify storage consistency: messages and chat_history before and after
    verify_storage_consistency(
        before_session, final_session,
        before_output, final_output,
        "Cross-Process Durable Execution - Error to Completion"
    )
    
    # Verify current state consistency
    verify_session_messages_consistency(final_session, final_output, "Final - Storage")
    
    print("\n[STEP 4] PASSED")
    
    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Durable Cross-Process")
    print("="*80)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run all comprehensive tests."""
    
    await test_external_tool_comprehensive()
    await test_cancel_run_comprehensive()
    await test_durable_execution_comprehensive()
    await test_durable_cross_process_comprehensive()
    
    print("\n" + "="*80)
    print("ALL COMPREHENSIVE TESTS PASSED!")
    print("  - External Tool: PASSED")
    print("  - Cancel Run: PASSED")
    print("  - Durable Execution: PASSED")
    print("  - Durable Cross-Process: PASSED")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
