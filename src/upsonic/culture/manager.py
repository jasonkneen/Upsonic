"""
CultureManager for handling user-provided cultural knowledge.

This manager handles:
- Accepting user-provided CulturalKnowledge instances or string descriptions
- Using an Agent to create/refine cultural knowledge based on user input
- Combining user input with stored cultural knowledge
- Formatting cultural knowledge for system prompt injection

Storage operations are handled by the Memory class, not CultureManager.

"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    from upsonic.models import Model

# System prompt for the cultural knowledge extraction agent
CULTURE_EXTRACTION_SYSTEM_PROMPT = """You are a Cultural Knowledge Architect responsible for extracting and refining cultural knowledge from user inputs.

Your role is to:
1. Analyze user-provided descriptions, instructions, or context
2. Extract meaningful cultural principles, guidelines, and best practices
3. Create or update CulturalKnowledge entries that capture:
   - Name: A concise, specific title for the knowledge
   - Summary: A one-line purpose or takeaway
   - Content: Detailed principles, rules, or guidelines
   - Categories: Relevant tags (e.g., 'engineering', 'communication', 'principles')
   - Notes: Contextual notes, rationale, or examples

Guidelines for extraction:
- Focus on UNIVERSAL principles that benefit all agents, not user-specific preferences
- Extract actionable guidelines, not just observations
- Preserve the intent and context of the original input
- Be specific and concrete, avoid vague generalizations
- If updating existing knowledge, MERGE and ENHANCE rather than replace
- Categories should be lowercase, hyphenated (e.g., 'code-review', 'customer-service')

When given existing cultural knowledge:
- Review and consider what's already captured
- Add new insights without duplicating existing content
- Update if new information contradicts or improves existing knowledge
- Maintain consistency in naming and categorization
"""


class CultureManager:
    """Manager for cultural knowledge user input and formatting.
    
    CultureManager handles:
    1. Accepting user-provided CulturalKnowledge or string descriptions
    2. Using an Agent to create/refine cultural knowledge from strings
    3. Combining user input with stored cultural knowledge
    4. Formatting for system prompt injection
    
    Storage operations are managed by the Memory class, NOT CultureManager.
    
    Usage:
        # With explicit CulturalKnowledge
        knowledge = CulturalKnowledge(
            name="Code Review Standards",
            content="Focus on maintainability, security, and performance"
        )
        manager = CultureManager(model="openai/gpt-4o")
        manager.set_cultural_knowledge(knowledge)
        
        # With string description (Agent creates CulturalKnowledge)
        manager = CultureManager(model="openai/gpt-4o")
        manager.set_cultural_knowledge("I want my agent to be helpful and professional")
        await manager.aprepare()  # This processes the string input
    
    """
    
    def __init__(
        self,
        model: Optional[Union["Model", str]] = None,
        enabled: bool = True,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        debug: bool = False,
        debug_level: int = 1,
    ) -> None:
        """
        Initialize the CultureManager.
        
        Args:
            model: Model for cultural knowledge extraction
            enabled: Whether culture management is enabled
            agent_id: Agent ID for culture context
            team_id: Team ID for culture context
            debug: Enable debug logging
            debug_level: Debug verbosity level (1-3)
        """
        self._model_spec = model
        self.enabled = enabled
        self.agent_id = agent_id
        self.team_id = team_id
        self.debug = debug
        self.debug_level = debug_level
        
        # Current cultural knowledge (user-provided or extracted)
        self._cultural_knowledge: Optional["CulturalKnowledge"] = None
        
        # Raw string input that needs processing (set by set_cultural_knowledge)
        self._pending_string_input: Optional[str] = None
        
        # Whether the user input was a CulturalKnowledge instance (no processing needed)
        self._is_instance_input: bool = False
        
        # Cultural knowledge loaded from storage (set by MemoryManager)
        self._stored_knowledge: List["CulturalKnowledge"] = []
        
        # Track if knowledge was updated in this session
        self._knowledge_updated: bool = False
        
        # Track if aprepare was called
        self._prepared: bool = False
    
    @property
    def cultural_knowledge(self) -> Optional["CulturalKnowledge"]:
        """Get the current cultural knowledge."""
        return self._cultural_knowledge
    
    @property
    def stored_knowledge(self) -> List["CulturalKnowledge"]:
        """Get cultural knowledge loaded from storage."""
        return self._stored_knowledge
    
    @stored_knowledge.setter
    def stored_knowledge(self, value: List["CulturalKnowledge"]) -> None:
        """Set stored knowledge (used by MemoryManager)."""
        self._stored_knowledge = value
    
    @property
    def knowledge_updated(self) -> bool:
        """Check if knowledge was updated in this session."""
        return self._knowledge_updated
    
    @property
    def has_pending_input(self) -> bool:
        """Check if there's a pending string input that needs processing."""
        return self._pending_string_input is not None and not self._prepared
    
    def set_cultural_knowledge(
        self,
        knowledge: Union["CulturalKnowledge", str],
    ) -> None:
        """
        Set cultural knowledge from user input.
        
        If a string is provided, it will be stored for later processing
        by aprepare() which uses an Agent to extract structured knowledge.
        
        If a CulturalKnowledge instance is provided, it's used directly.
        
        Args:
            knowledge: CulturalKnowledge instance or string description
        """
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        if isinstance(knowledge, str):
            # Store the string for later processing in aprepare()
            self._pending_string_input = knowledge
            self._is_instance_input = False
            # Also create a basic fallback in case aprepare() isn't called
            self._cultural_knowledge = CulturalKnowledge(
                id=str(uuid.uuid4()),
                name="User Cultural Guidelines",
                content=knowledge,
                summary="User-provided cultural guidelines",
                categories=["user-provided"],
            )
        else:
            # Use provided CulturalKnowledge instance directly
            if knowledge.id is None:
                knowledge.id = str(uuid.uuid4())
            self._cultural_knowledge = knowledge
            self._pending_string_input = None
            self._is_instance_input = True
        
        self._knowledge_updated = True
    
    async def aprepare(self) -> None:
        """
        Prepare cultural knowledge by processing any pending string input.
        
        This method should be called after stored_knowledge is set by MemoryManager.
        
        If user provided a string:
        - Calls acreate_cultural_knowledge() with stored knowledge as context
        - Creates a properly structured CulturalKnowledge instance
        
        If user provided a CulturalKnowledge instance:
        - No processing needed, uses directly
        
        After this, format_for_system_prompt() will combine user input + stored knowledge.
        """
        if self._prepared:
            return
        
        self._prepared = True
        
        # If user provided a string, process it with an Agent
        if self._pending_string_input and not self._is_instance_input:
            await self._process_string_input(self._pending_string_input)
    
    def prepare(self) -> None:
        """Synchronous version of aprepare."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(asyncio.run, self.aprepare()).result()
        except RuntimeError:
            asyncio.run(self.aprepare())
    
    async def _process_string_input(self, message: str) -> Optional["CulturalKnowledge"]:
        """
        Process a string input into structured CulturalKnowledge using an Agent.
        
        This uses stored_knowledge as context for the extraction.
        
        Args:
            message: User message describing desired cultural behaviors
            
        Returns:
            Created CulturalKnowledge instance
        """
        from pydantic import BaseModel, Field
        from typing import List as ListType, Optional as OptionalType
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        from upsonic.utils.printing import info_log, warning_log
        
        if not self._model_spec:
            if self.debug:
                warning_log(
                    "CultureManager: No model configured, using basic extraction",
                    "CultureManager"
                )
            # Keep the basic fallback that was already created
            return self._cultural_knowledge
        
        # Define output schema for extraction
        class ExtractedCulture(BaseModel):
            """Extracted cultural knowledge from user input."""
            name: str = Field(..., description="Concise, specific title for the knowledge")
            summary: str = Field(..., description="One-line purpose or takeaway")
            content: str = Field(..., description="Detailed principles, rules, or guidelines")
            categories: ListType[str] = Field(
                default_factory=list,
                description="Relevant tags (lowercase, hyphenated)"
            )
            notes: OptionalType[ListType[str]] = Field(
                default=None,
                description="Contextual notes, rationale, or examples"
            )
        
        # Build context with existing stored knowledge
        context_parts: List[str] = []
        
        if self._stored_knowledge:
            existing_str = "\n".join([
                f"- {k.name}: {k.summary or ''}" 
                for k in self._stored_knowledge[:5]
            ])
            context_parts.append(f"Existing Cultural Knowledge from Storage:\n{existing_str}")
        
        context_str = "\n\n".join(context_parts) if context_parts else "No existing cultural knowledge in storage."
        
        # Create extraction task
        extraction_prompt = f"""Analyze the following user input and extract cultural knowledge.

{context_str}

User Input:
{message}

Extract the cultural knowledge as structured data. Focus on universal principles that benefit all agents.
Consider the existing knowledge above and create complementary knowledge that enhances the overall culture.
"""
        
        try:
            from upsonic.agent.agent import Agent
            from upsonic.tasks.tasks import Task
            
            extractor = Agent(
                model=self._model_spec,
                name="Culture Extractor",
                system_prompt=CULTURE_EXTRACTION_SYSTEM_PROMPT,
                debug=self.debug,
            )
            
            task = Task(
                description=extraction_prompt,
                response_format=ExtractedCulture,
            )
            
            result = await extractor.do_async(task)
            
            if result and hasattr(result, 'name'):
                # Create CulturalKnowledge from extracted data
                self._cultural_knowledge = CulturalKnowledge(
                    id=str(uuid.uuid4()),
                    name=result.name,
                    summary=result.summary,
                    content=result.content,
                    categories=result.categories,
                    notes=result.notes,
                    input=message,
                    agent_id=self.agent_id,
                    team_id=self.team_id,
                )
                
                self._knowledge_updated = True
                
                if self.debug:
                    info_log(
                        f"Extracted cultural knowledge: {self._cultural_knowledge.name}",
                        "CultureManager"
                    )
                
                return self._cultural_knowledge
            else:
                if self.debug:
                    warning_log("Culture extraction returned no result, using basic fallback", "CultureManager")
                # Keep the basic fallback
                return self._cultural_knowledge
                
        except Exception as e:
            if self.debug:
                warning_log(f"Culture extraction failed: {e}, using basic fallback", "CultureManager")
            # Keep the basic fallback
            return self._cultural_knowledge
    
    async def acreate_cultural_knowledge(
        self,
        message: str,
        existing_knowledge: Optional[List["CulturalKnowledge"]] = None,
    ) -> Optional["CulturalKnowledge"]:
        """
        Create cultural knowledge from a user message using an Agent.
        
        This method uses an LLM to analyze the user's message and extract
        meaningful cultural knowledge that can be applied across agents.
        
        Args:
            message: User message describing desired cultural behaviors
            existing_knowledge: Optional list of existing knowledge for context
            
        Returns:
            Created CulturalKnowledge instance
        """
        # If existing_knowledge is provided, temporarily set it as stored_knowledge
        if existing_knowledge:
            self._stored_knowledge = existing_knowledge
        
        self._pending_string_input = message
        self._is_instance_input = False
        
        return await self._process_string_input(message)
    
    def create_cultural_knowledge(
        self,
        message: str,
        existing_knowledge: Optional[List["CulturalKnowledge"]] = None,
    ) -> Optional["CulturalKnowledge"]:
        """Synchronous version of acreate_cultural_knowledge."""
        import asyncio
        try:
            _ = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run,
                    self.acreate_cultural_knowledge(message, existing_knowledge)
                ).result()
        except RuntimeError:
            return asyncio.run(
                self.acreate_cultural_knowledge(message, existing_knowledge)
            )
    
    def get_combined_knowledge(self) -> List["CulturalKnowledge"]:
        """
        Get all cultural knowledge (user-provided + stored) for system prompt.
        
        ALWAYS combines user-provided knowledge with stored knowledge:
        1. User-provided knowledge (string or instance) takes priority (listed first)
        2. Stored knowledge is added after, avoiding duplicates by ID
        
        If no user input was provided but there's stored knowledge,
        the stored knowledge is still returned.
        
        Returns:
            Combined list of CulturalKnowledge instances
        """
        combined: List["CulturalKnowledge"] = []
        
        # Add user-provided knowledge first (higher priority)
        if self._cultural_knowledge:
            combined.append(self._cultural_knowledge)
        
        # Add stored knowledge (avoid duplicates by ID)
        seen_ids = {k.id for k in combined if k.id}
        for knowledge in self._stored_knowledge:
            if knowledge.id and knowledge.id not in seen_ids:
                combined.append(knowledge)
                seen_ids.add(knowledge.id)
        
        return combined
    
    def format_for_system_prompt(
        self,
        max_length: int = 3000,
    ) -> Optional[str]:
        """
        Format cultural knowledge for system prompt injection.
        
        Combines user-provided cultural knowledge with stored cultural knowledge
        and formats it for injection into the system prompt.
        
        Args:
            max_length: Maximum length of formatted output
            
        Returns:
            Formatted string for system prompt, or None if empty
        """
        combined = self.get_combined_knowledge()
        if not combined:
            return None
        
        parts: List[str] = []
        current_length = 0
        
        for knowledge in combined:
            entry_parts: List[str] = []
            
            if knowledge.name:
                entry_parts.append(f"### {knowledge.name}")
            
            if knowledge.summary:
                entry_parts.append(f"**Purpose:** {knowledge.summary}")
            
            if knowledge.content:
                entry_parts.append(f"\n{knowledge.content}")
            
            if knowledge.categories:
                entry_parts.append(f"\n*Tags: {', '.join(knowledge.categories)}*")
            
            if knowledge.notes:
                notes_str = "\n".join(f"- {note}" for note in knowledge.notes[:3])
                entry_parts.append(f"\n**Notes:**\n{notes_str}")
            
            entry = "\n".join(entry_parts)
            entry_length = len(entry)
            
            if current_length + entry_length > max_length:
                break
            
            parts.append(entry)
            current_length += entry_length + 4  # +4 for separators
        
        if not parts:
            return None
        
        formatted = "\n\n---\n\n".join(parts)
        
        instruction = (
            "**Important:** You are not required to use all of the cultural knowledge provided below. "
            "Select and apply only the cultural knowledge that is relevant and useful for the current task. "
            "You may use one, multiple, or none of the cultural knowledge entries based on what best serves the task at hand.\n\n"
        )
        
        return f"<CulturalKnowledge>\n{instruction}{formatted}\n</CulturalKnowledge>"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize CultureManager state to dictionary.
        
        Returns:
            Dictionary representation of the manager state
        """
        result: Dict[str, Any] = {
            "enabled": self.enabled,
            "agent_id": self.agent_id,
            "team_id": self.team_id,
            "knowledge_updated": self._knowledge_updated,
            "prepared": self._prepared,
        }
        
        if self._cultural_knowledge:
            result["cultural_knowledge"] = self._cultural_knowledge.to_dict()
        else:
            result["cultural_knowledge"] = None
        
        if self._stored_knowledge:
            result["stored_knowledge"] = [k.to_dict() for k in self._stored_knowledge]
        else:
            result["stored_knowledge"] = []
        
        return result
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        model: Optional[Union["Model", str]] = None,
    ) -> "CultureManager":
        """
        Create CultureManager from dictionary.
        
        Args:
            data: Dictionary containing manager state
            model: Model for cultural knowledge extraction
            
        Returns:
            CultureManager instance
        """
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        manager = cls(
            model=model,
            enabled=data.get("enabled", True),
            agent_id=data.get("agent_id"),
            team_id=data.get("team_id"),
        )
        
        manager._knowledge_updated = data.get("knowledge_updated", False)
        manager._prepared = data.get("prepared", False)
        
        knowledge_data = data.get("cultural_knowledge")
        if knowledge_data and isinstance(knowledge_data, dict):
            manager._cultural_knowledge = CulturalKnowledge.from_dict(knowledge_data)
        
        stored_data = data.get("stored_knowledge", [])
        if stored_data:
            manager._stored_knowledge = [
                CulturalKnowledge.from_dict(k) if isinstance(k, dict) else k
                for k in stored_data
            ]
        
        return manager
