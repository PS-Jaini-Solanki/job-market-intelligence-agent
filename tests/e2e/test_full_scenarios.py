import pytest
from agents.state import create_initial_state

def test_create_initial_state():
    state = create_initial_state(user_query="Test query", country_filter="US")
    assert state["user_query"] == "Test query"
    assert state["country_filter"] == "US"
    assert state["execution_plan"] is None
    assert state["should_stop"] is False
