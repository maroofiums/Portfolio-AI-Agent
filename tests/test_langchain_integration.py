from app.main import ChatRequest, Message, build_langchain_messages


def test_build_langchain_messages_converts_roles():
    payload = [
        Message(role="user", content="Hello there"),
        Message(role="assistant", content="Hi!"),
    ]

    result = build_langchain_messages(payload)

    assert len(result) == 2
    assert result[0].content == "Hello there"
    assert result[1].content == "Hi!"
    assert result[0].type == "human"
    assert result[1].type == "ai"


def test_chat_request_accepts_minimal_payload():
    request = ChatRequest(messages=[Message(role="user", content="Tell me about your work")])

    assert request.messages[0].content == "Tell me about your work"
