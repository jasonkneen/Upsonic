import pytest
from upsonic import Agent, Task


def test_print_do_default_prints():
    """Test: print_do should print by default"""
    print("test_print_do_default_prints")
    my_agent = Agent()
    task = Task("test task")
    my_agent.print_do(task)


def test_do_default_no_prints():
    """Test: do should not print by default"""
    print("test_do_default_no_prints")
    my_agent = Agent()
    task = Task("test task")
    my_agent.do(task)


def test_print_do_with_print_false():
    """Test: print_do should not print when print=False"""
    print("test_print_do_with_print_false")
    my_agent = Agent(print=False)
    task = Task("test task")
    my_agent.print_do(task)


def test_do_with_print_true():
    """Test: do should print when print=True"""
    print("test_do_with_print_true")
    my_agent = Agent(print=True)
    task = Task("test task")
    my_agent.do(task)


def test_do_with_env_print_true(monkeypatch: pytest.MonkeyPatch):
    """Test: do should print when ENV UPSONIC_AGENT_PRINT=true"""
    print("test_do_with_env_print_true")
    monkeypatch.setenv("UPSONIC_AGENT_PRINT", "true")
    my_agent = Agent()
    task = Task("test task")
    my_agent.do(task)


def test_do_with_env_print_false_and_agent_print_true(monkeypatch: pytest.MonkeyPatch):
    """Test: do should not print when ENV=false but agent print=True (ENV overrides)"""
    print("test_do_with_env_print_false_and_agent_print_true")
    monkeypatch.setenv("UPSONIC_AGENT_PRINT", "false")
    my_agent = Agent(print=True)
    task = Task("test task")
    my_agent.do(task)


def test_print_do_with_env_print_false(monkeypatch: pytest.MonkeyPatch):
    """Test: print_do should not print when ENV UPSONIC_AGENT_PRINT=false"""
    print("test_print_do_with_env_print_false")
    monkeypatch.setenv("UPSONIC_AGENT_PRINT", "false")
    my_agent = Agent()
    task = Task("test task")
    my_agent.print_do(task)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])