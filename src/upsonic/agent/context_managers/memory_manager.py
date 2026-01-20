from __future__ import annotations
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import asyncio


if TYPE_CHECKING:
    from upsonic.storage.memory.memory import Memory
    from upsonic.run.agent.output import AgentRunOutput
    from upsonic.culture.manager import CultureManager


class MemoryManager:
    """
    A context manager that integrates the Memory orchestrator into the agent's
    execution pipeline.

    This manager is responsible for:
    1. Preparing memory inputs before the LLM call (prepare_inputs_for_task)
    2. Combining user-provided cultural knowledge with stored cultural knowledge
    3. Saving the session after the run completes or pauses (save_session_async)
    """

    def __init__(
        self,
        memory: Optional["Memory"],
        agent_metadata: Optional[Dict[str, Any]] = None,
        culture_manager: Optional["CultureManager"] = None,
    ):
        """
        Initializes the MemoryManager.

        Args:
            memory: The configured Memory object from the parent agent.
            agent_metadata: Optional metadata dict from the Agent to inject into prompts.
            culture_manager: Optional CultureManager for cultural knowledge handling.
        """
        self.memory = memory
        self.agent_metadata = agent_metadata or {}
        self._culture_manager = culture_manager
        self._prepared_inputs: Dict[str, Any] = {
            "message_history": [],
            "context_injection": "",
            "system_prompt_injection": "",
            "metadata_injection": "",
            "culture_injection": "",
            "cultural_knowledge_list": [],
        }
        self._agent_run_output: Optional["AgentRunOutput"] = None

    def get_message_history(self) -> List[Any]:
        """
        Provides the prepared message history (full session memory) to the
        agent's core run method.
        """
        return self._prepared_inputs.get("message_history", [])

    def get_context_injection(self) -> str:
        """
        Provides the prepared context string (e.g., session summary) to the
        ContextManager.
        """
        return self._prepared_inputs.get("context_injection", "")

    def get_system_prompt_injection(self) -> str:
        """
        Provides the prepared system prompt string (e.g., user profile) to
        the SystemPromptManager.
        """
        return self._prepared_inputs.get("system_prompt_injection", "")
    
    def get_metadata_injection(self) -> str:
        """
        Provides the prepared metadata string to inject into the user prompt.
        This includes both agent-level metadata and session-level metadata.
        """
        return self._prepared_inputs.get("metadata_injection", "")
    
    def get_culture_injection(self) -> str:
        """
        Provides the formatted cultural knowledge string for system prompt injection.
        
        This combines:
        - User-provided cultural knowledge (from Agent parameter)
        - Cultural knowledge from storage (loaded via Memory)
        
        Uses CultureManager.format_for_system_prompt() to combine and format.
        
        Returns:
            Formatted cultural knowledge string, or empty string if not available.
        """
        # If we have a CultureManager, use it to get the combined & formatted culture
        if self._culture_manager:
            formatted = self._culture_manager.format_for_system_prompt()
            return formatted or ""
        
        # Fallback to raw culture injection from memory
        return self._prepared_inputs.get("culture_injection", "")
    
    def get_cultural_knowledges(self) -> List[Any]:
        """
        Provides the list of CulturalKnowledge instances loaded from storage.
        
        Returns:
            List of CulturalKnowledge instances.
        """
        return self._prepared_inputs.get("cultural_knowledge_list", [])
    
    @property
    def culture_manager(self) -> Optional["CultureManager"]:
        """Get the CultureManager instance."""
        return self._culture_manager

    def set_run_output(self, run_output: "AgentRunOutput") -> None:
        """
        Set the AgentRunOutput for session save.
        
        Args:
            run_output: The AgentRunOutput to save
        """
        self._agent_run_output = run_output

    async def aprepare(self) -> None:
        """
        Prepare memory inputs before the LLM call.
        
        This method prepares message history, context injection, system prompt 
        injection, metadata injection, and cultural knowledge from the memory module.
        
        If a CultureManager is provided:
        1. Sets stored cultural knowledge from Memory on CultureManager
        2. Calls CultureManager.aprepare() to process any pending string input:
           - If user provided STRING: Uses Agent to extract structured CulturalKnowledge
           - If user provided CulturalKnowledge instance: Uses directly
        3. CultureManager.format_for_system_prompt() then combines user input + stored knowledge
        """
        if self.memory:
            self._prepared_inputs = await self.memory.prepare_inputs_for_task(
                agent_metadata=self.agent_metadata
            )
            
            # If we have a CultureManager, set stored knowledge and prepare it
            if self._culture_manager:
                # Step 1: Set stored knowledge from Memory onto CultureManager
                stored_knowledge = self._prepared_inputs.get("cultural_knowledge_list", [])
                if stored_knowledge:
                    self._culture_manager.stored_knowledge = stored_knowledge
                
                # Step 2: Process any pending string input
                # If user provided string to Agent(cultural_knowledge="..."),
                # this calls an Agent to extract structured CulturalKnowledge
                # using stored knowledge as context
                await self._culture_manager.aprepare()
        else:
            # No memory, but we may still have a CultureManager with user input
            if self._culture_manager:
                # Process pending string input even without stored knowledge
                await self._culture_manager.aprepare()
            
            if self.agent_metadata:
                metadata_parts = []
                for key, value in self.agent_metadata.items():
                    metadata_parts.append(f"  {key}: {value}")
                if metadata_parts:
                    self._prepared_inputs["metadata_injection"] = (
                        "<AgentMetadata>\n" + "\n".join(metadata_parts) + "\n</AgentMetadata>"
                    )
    
    async def afinalize(self) -> None:
        """
        Finalize and save session after the run.
        
        This method saves the session to storage via Memory.save_session_async().
        """
        if self.memory and self._agent_run_output:
            await self.memory.save_session_async(
                output=self._agent_run_output,
            )
    
    def prepare(self) -> None:
        """Synchronous version of aprepare."""
        asyncio.get_event_loop().run_until_complete(self.aprepare())
    
    def finalize(self) -> None:
        """Synchronous version of afinalize."""
        asyncio.get_event_loop().run_until_complete(self.afinalize())

    @asynccontextmanager
    async def manage_memory(self):
        """
        The asynchronous context manager for orchestrating memory operations
        throughout a task's lifecycle.
        
        Note: This context manager is kept for backward compatibility.
        For step-based architecture, use aprepare() and afinalize() directly.
        """
        await self.aprepare()
        
        try:
            yield self
        finally:
            await self.afinalize()
