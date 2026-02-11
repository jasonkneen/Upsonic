"""
Smoke test for Team streaming in sequential mode with mixed Agent and Team entities.
Verifies stream() yields non-empty text and entity headers appear in output.
"""

import pytest
from upsonic import Agent, Task, Team

pytestmark = pytest.mark.timeout(120)


def test_sequential_streaming_mixed_entities() -> None:
    """Sequential mode: stream with one Agent and one nested Team; output has headers and content."""
    researcher = Agent(
        model="openai/gpt-4o",
        name="Researcher",
        role="Research Specialist",
        goal="Find accurate information and data",
    )
    editor = Agent(
        model="openai/gpt-4o",
        name="Editor",
        role="Editor",
        goal="Polish and refine content",
    )
    writer_team = Team(
        entities=[
            Agent(
                model="openai/gpt-4o",
                name="Writer",
                role="Content Writer",
                goal="Create clear and engaging content",
            ),
            editor,
        ],
        mode="sequential",
        name="WriterTeam",
    )
    team = Team(
        entities=[researcher, writer_team],
        mode="sequential",
    )
    tasks = [
        Task(description="Research the latest developments in quantum computing"),
        Task(description="Write a blog post about quantum computing for general audience"),
    ]
    chunks: list[str] = []
    for chunk in team.stream(tasks):
        chunks.append(chunk)
    output = "".join(chunks)
    assert output, "Stream output should be non-empty"
    assert "--- [" in output, "Entity stream headers (--- [...) should appear in output"
