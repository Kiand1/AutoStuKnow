import json
from pathlib import Path

import respx
from fastapi.testclient import TestClient
from httpx import Response

from app.config import Settings
from app.main import create_app

API_KEY = "test-key-that-is-at-least-24-characters"
MCP_KEY = "test-mcp-key-that-is-at-least-24-characters"
INITIAL_WEB_PASSWORD = "test-web-password-123456789"
USER_WEB_PASSWORD = "memorable-password-123"


def build_settings(tmp_path: Path, **updates: object) -> Settings:
    values: dict[str, object] = {
        "data_dir": tmp_path,
        "ingestor_api_key": API_KEY,
        "mcp_api_key": MCP_KEY,
        "mcp_enabled": False,
        "web_ui_username": "admin",
        "web_ui_password": INITIAL_WEB_PASSWORD,
        "web_ui_session_secret": "test-web-session-secret-that-is-at-least-32-characters",
        "anythingllm_auto_sync": False,
    }
    values.update(updates)
    return Settings(**values)


def authenticate_web(client: TestClient, password: str = INITIAL_WEB_PASSWORD) -> None:
    login = client.post(
        "/ui/api/login",
        json={"username": "admin", "password": password},
    )
    assert login.status_code == 200
    if login.json()["must_change_password"]:
        changed = client.post(
            "/ui/api/password",
            json={"new_password": USER_WEB_PASSWORD},
        )
        assert changed.status_code == 200


def mcp_headers(token: str = MCP_KEY) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }


def test_web_switch_controls_authenticated_mcp_and_persists(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        disabled = client.post(
            "/mcp",
            headers=mcp_headers(),
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        unauthenticated_settings = client.get("/ui/api/mcp")
        authenticate_web(client)
        initial = client.get("/ui/api/mcp")
        enabled = client.post("/ui/api/mcp", json={"enabled": True})

        missing_token = client.post(
            "/mcp",
            headers={"Accept": "application/json, text/event-stream"},
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        wrong_token = client.post(
            "/mcp",
            headers=mcp_headers("wrong-token-that-is-long-enough"),
            json={"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}},
        )
        tools = client.post(
            "/mcp",
            headers=mcp_headers(),
            json={"jsonrpc": "2.0", "id": 4, "method": "tools/list", "params": {}},
        )

    assert disabled.status_code == 503
    assert unauthenticated_settings.status_code == 401
    assert initial.json()["enabled"] is False
    assert initial.json()["endpoint"] == "http://testserver/mcp"
    assert enabled.json()["enabled"] is True
    assert missing_token.status_code == 401
    assert wrong_token.status_code == 401
    assert tools.status_code == 200
    listed_tools = tools.json()["result"]["tools"]
    assert {item["name"] for item in listed_tools} == {
        "list_workspaces",
        "list_knowledge_tree",
        "search_knowledge",
        "get_knowledge",
    }
    assert all(item["annotations"]["readOnlyHint"] is True for item in listed_tools)
    assert all(item["annotations"]["destructiveHint"] is False for item in listed_tools)

    restarted_app = create_app(build_settings(tmp_path))
    with TestClient(restarted_app) as restarted:
        authenticate_web(restarted, USER_WEB_PASSWORD)
        persisted = restarted.get("/ui/api/mcp")
    assert persisted.json()["enabled"] is True


@respx.mock
def test_search_knowledge_uses_anythingllm_vectors_and_directory_filter(
    tmp_path: Path,
) -> None:
    base_url = "http://anythingllm.test/api"
    settings = build_settings(
        tmp_path,
        mcp_enabled=True,
        anythingllm_base_url=base_url,
        anythingllm_api_key="anythingllm-test-key",
    )
    respx.get(f"{base_url}/v1/workspace/research").mock(
        return_value=Response(
            200,
            json={
                "workspace": [
                    {"id": 1, "name": "研究", "slug": "research", "documents": []}
                ]
            },
        )
    )
    vector_route = respx.post(f"{base_url}/v1/workspace/research/vector-search").mock(
        return_value=Response(
            200,
            json={
                "results": [
                    {
                        "id": "matching",
                        "text": "MA20 回踩后等待确认信号。",
                        "score": 0.91,
                        "distance": 0.09,
                        "metadata": {
                            "title": "双均线入场规则",
                            "docSource": "https://youtu.be/example",
                            "description": (
                                "YouTube transcript imported by AutoStuKnow; "
                                "directory: 投资/虚拟币"
                            ),
                        },
                    },
                    {
                        "id": "filtered-out",
                        "text": "股票知识",
                        "score": 0.8,
                        "metadata": {
                            "title": "股票",
                            "description": (
                                "YouTube transcript imported by AutoStuKnow; directory: 投资/股票"
                            ),
                        },
                    },
                ]
            },
        )
    )

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/mcp",
            headers=mcp_headers(),
            json={
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "search_knowledge",
                    "arguments": {
                        "workspace_slug": "research",
                        "query": "MA20 如何开仓",
                        "category_path": "投资/虚拟币",
                        "top_k": 3,
                    },
                },
            },
        )

    assert response.status_code == 200
    result = response.json()["result"]["structuredContent"]
    assert result["count"] == 1
    assert result["results"][0]["title"] == "双均线入场规则"
    assert result["results"][0]["source_url"] == "https://youtu.be/example"
    assert result["results"][0]["category_path"] == "投资/虚拟币"
    request_payload = json.loads(vector_route.calls[0].request.read())
    assert request_payload["topN"] == 15
