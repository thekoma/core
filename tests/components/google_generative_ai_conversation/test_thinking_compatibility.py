"""Tests for the Google Generative AI Conversation integration with Thinking models."""

from unittest.mock import AsyncMock, patch

from google.genai.types import GenerateContentResponse
import pytest
from syrupy.assertion import SnapshotAssertion

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent

from tests.common import MockConfigEntry
from tests.components.conversation import (
    MockChatLog,
    mock_chat_log,  # noqa: F401
)


@pytest.fixture(autouse=True)
def mock_ulid_tools():
    """Mock generated ULIDs for tool calls."""
    with patch("homeassistant.helpers.llm.ulid_now", return_value="mock-tool-call"):
        yield


@pytest.mark.usefixtures("mock_init_component")
@pytest.mark.usefixtures("mock_ulid_tools")
async def test_thinking_model_empty_text_with_function_call(
    hass: HomeAssistant,
    mock_config_entry_with_assist: MockConfigEntry,
    mock_chat_log: MockChatLog,  # noqa: F811
    mock_send_message_stream: AsyncMock,
    snapshot: SnapshotAssertion,
) -> None:
    """Test that empty text parts alongside function calls are ignored."""
    agent_id = "conversation.google_ai_conversation"
    context = Context()

    messages = [
        # Function call stream with empty text part (The Scenario A)
        [
            GenerateContentResponse(
                candidates=[
                    {
                        "content": {
                            "parts": [
                                {
                                    "function_call": {
                                        "name": "test_tool",
                                        "args": {
                                            "param1": "test_value",
                                        },
                                    },
                                    "thought_signature": b"_thought_signature_1",
                                },
                                {
                                    "text": "",  # This is the problematic empty part
                                },
                            ],
                            "role": "model",
                        }
                    }
                ]
            ),
        ],
        # Follow-up after tool result (Response from model)
        [
            GenerateContentResponse(
                candidates=[
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "Done!",
                                }
                            ],
                            "role": "model",
                        },
                        "finish_reason": "STOP",
                    }
                ]
            ),
        ],
    ]

    mock_send_message_stream.return_value = messages

    mock_chat_log.mock_tool_results(
        {
            "mock-tool-call": {"result": "Success"},
        }
    )

    # 1. Start conversation, triggering tool call
    result = await conversation.async_converse(
        hass,
        "Do the thing",
        mock_chat_log.conversation_id,
        context,
        agent_id=agent_id,
        device_id="test_device",
    )

    # Verify action done (means tool execution loop completed)
    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert result.response.as_dict()["speech"]["plain"]["speech"] == "Done!"

    # 2. Verify history sent to Google in the second turn
    # It should contain:
    # - User: "Do the thing"
    # - Model: Function Call (and NO empty text)
    # - User (Tool Result): "Success"

    # We check what was passed to `send_message_stream` or `create` (history)
    # The `conversation.async_converse` calls `_async_handle_chat_log`.
    # `chat` object is created via `self._genai_client.aio.chats.create`.

    # We can inspect the history argument passed to create used for the *second* turn (if specific create calls are made per turn? No, it's one chat session usually, but let's see how the loop works).
    # Actually, `async_converse` in this integration creates a NEW chat session for every user turn?
    # Let's check `entity.py`: `chat = self._genai_client.aio.chats.create(...)` is called inside `_async_handle_chat_log`.
    # Yes, it creates a fresh chat session using `chat_log` history.

    # So we should be able to check the `history` passed to `create`.

    with patch(
        "google.genai.chats.AsyncChats.create", return_value=AsyncMock()
    ) as mock_create:
        mock_create.return_value.send_message_stream = mock_send_message_stream

        # We need to run a follow-up conversation to see what history is built from the previous turn's result
        # BUT, `async_converse` does the whole tool loop internally if I'm not mistaken?
        # `_async_handle_chat_log` loops `MAX_TOOL_ITERATIONS`.

        # In the test setup above `mock_send_message_stream` provided the sequence for the tool loop.
        # So `async_converse` should have completed the loop.

        # To verify the history construction logic, we can inspect what was passed to `chat.send_message_stream`?
        # No, `chat.send_message_stream` takes the *new* message (User input or Tool Result).
        # The history is passed to `chats.create`.

        # In the loop:
        # 1. `chats.create(history=[User: "Do something"])`
        # 2. `chat.send_message_stream("Do something")` -> Returns [Model: Call + Empty Text]
        # 3. Code adds responses to `chat_log`.
        # 4. Code generates tool result.
        # 5. `chat_request` is prepared with Tool Result.
        # 6. `chat.send_message_stream(Tool Result)`

        # Validating step 6: The `chat` object's internal history would theoretically now contain [Model: Call + Empty Text].
        # However, the Google Python SDK handles the history locally in the `chat` object.
        # We are mocking the SDK.
        # The critical part is how HA converts its `chat_log` (which is HA's internal memory) back to Google `Content` objects if we were to start a NEW turn/conversation or if we rely on `chat_log` for persistence.

        # HA `conversation` integration persists state in `chat_log`.
        # The integration reconstructs the history from `chat_log` every time `_async_handle_chat_log` is called?
        # No, `_async_handle_chat_log` takes `chat_log` as input.

        # So if we simply verify that `chat_log` does NOT contain the empty text message, we are good?
        # Let's verify what `chat_log` contains after the interaction.

    # Let's verify the tool call arguments
    # Because `mock_chat_log` is updated in place

    # We want to check that we don't have a standalone empty content message in chat_log
    # Iterating over chat_log.content

    # We expect:
    # 0. System prompt
    # 1. User: "Do the thing"
    # 2. Assistant: Tool Call (FunctionCall)
    # 3. Tool Result
    # 4. Assistant: "Done!"

    # If the bug was present, we might see:
    # 2. Assistant: Tool Call
    # 3. Assistant: "" (Empty Text)
    # 4. Tool Result

    # But wait, `_transform_stream` yields chunks.
    # If it yields a chunk for empty text, `chat_log` might append it.

    # Let's check the mock_chat_log content
    # Note: MockChatLog might not perfectly reflect the real ChatLog behavior if we don't use the real one,
    # but the test uses `mock_chat_log` fixture which usually behaves like a real one or is a real instance.
    # Looking at `test_conversation.py` imports:
    # `from tests.components.conversation import MockChatLog, mock_chat_log`
    # It seems to be a mock. But `conversation.async_converse` calls methods on it.

    assert (
        len(mock_chat_log.content) == 5
    )  # System, User, Assistant(Tool), ToolResult, Assistant(Final)
    # Wait:
    # 0: System
    # 1: User "Do the thing"
    # 2: Assistant (Tool Call)
    # 3: Tool Result "Success"
    # 4: Assistant "Done!"

    # So count should be 5.

    for entry in mock_chat_log.content:
        if entry.role == "assistant":
            # Check if any assistant message is just empty string and no tool calls
            assert not (
                entry.content == ""
                and not entry.tool_calls
                and not entry.thinking_content
            )
