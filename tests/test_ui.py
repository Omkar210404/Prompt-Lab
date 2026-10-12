"""The web app's server, called in-process (no sockets) with the mock provider."""

import asyncio

import httpx
import pytest
import yaml

from promptlab.ui import create_app

BASE = "http://127.0.0.1:8321"
# p1 does not ask for one word, so the scripted mock answers with a sentence and fails exact_match.
MOCK = {"id": "m1", "provider": "mock", "mock": {"rules": [
    {"when": {"prompt_lacks": "one word"}, "behave": "prose_wrapped", "prefix": "It is ", "suffix": "."},
]}}
EXP = "exp.yaml"


@pytest.fixture
def workspace(write_experiment, tmp_path):
    write_experiment({"models": [MOCK]})
    return tmp_path


@pytest.fixture
async def client(workspace):
    transport = httpx.ASGITransport(app=create_app(workspace))
    async with httpx.AsyncClient(transport=transport, base_url=BASE) as http:
        yield http


async def finish(client, job: dict) -> dict:
    for _ in range(500):
        if job["status"] != "running":
            return job
        await asyncio.sleep(0.01)
        job = (await client.get(f"/api/jobs/{job['id']}")).json()
    raise AssertionError("run did not finish")


async def run_all(client, **body) -> dict:
    response = await client.post("/api/runs", json={"experiment": EXP, **body})
    assert response.status_code == 200, response.text
    job = await finish(client, response.json())
    assert job["status"] == "done", job
    return job


# --- pages ---

async def test_index_is_the_app_with_inputs_and_run_buttons(client):
    page = (await client.get("/")).text
    assert "<title>prompt-lab</title>" in page
    for control in ('id="experiment"', 'id="prompts"', 'id="try"', "Test this example", 'id="run-all"', "Run the full test"):
        assert control in page
    assert (await client.get("/favicon.svg")).headers["content-type"].startswith("image/svg+xml")
    assert (await client.get("/api/status")).json()["version"]


async def test_app_page_is_self_contained(client):
    page = (await client.get("/")).text
    assert "https://" not in page and "http://" not in page and "@import" not in page


# --- experiments ---

async def test_lists_experiments_and_skips_other_yaml(client, workspace):
    (workspace / "prices.yaml").write_text("m1: {input: 1, output: 2}\n", encoding="utf-8")
    (workspace / "broken.yaml").write_text("name: broken\nprompts: []\n", encoding="utf-8")
    (workspace / "examples").mkdir()
    (workspace / "examples" / "more.yaml").write_text(
        (workspace / EXP).read_text(encoding="utf-8").replace("cases.jsonl", "../cases.jsonl"), encoding="utf-8")

    listed = {item["id"]: item for item in (await client.get("/api/experiments")).json()}
    assert set(listed) == {"exp.yaml", "broken.yaml", "examples/more.yaml"}  # the price table is not an experiment
    assert listed["exp.yaml"] == {"id": "exp.yaml", "name": "test-exp", "description": "", "cases": 3, "mock": True}
    assert "error" in listed["broken.yaml"]


async def test_experiment_detail(client):
    detail = (await client.get("/api/experiment", params={"id": EXP})).json()
    assert detail["name"] == "test-exp" and len(detail["config_hash"]) == 64
    assert [p["id"] for p in detail["prompts"]] == ["p1", "p2"]
    assert detail["variables"] == ["text"]
    assert detail["models"] == [{"id": "m1", "provider": "mock", "model": "m1", "available": True, "problem": None}]
    assert detail["scorers"] == [{"name": "exact_match", "type": "exact_match", "required": True}]
    assert detail["dataset"]["cases"] == 3
    assert detail["cases"][0] == {"id": "c1", "vars": {"text": "great product"}, "expected": "positive", "tags": ["easy"]}


async def test_model_without_its_key_is_reported_unavailable(write_experiment, tmp_path, monkeypatch):
    monkeypatch.delenv("UI_TEST_KEY", raising=False)
    real = {"id": "real", "provider": "openai_compatible", "base_url": "https://llm.example/v1", "api_key_env": "UI_TEST_KEY"}
    write_experiment({"models": [MOCK, real]})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(tmp_path)), base_url=BASE) as http:
        models = (await http.get("/api/experiment", params={"id": EXP})).json()["models"]
        assert models[0]["available"] is True
        assert models[1]["available"] is False and "UI_TEST_KEY is not set" in models[1]["problem"]

        monkeypatch.setenv("UI_TEST_KEY", "sk-test")
        models = (await http.get("/api/experiment", params={"id": EXP})).json()["models"]
        assert models[1]["available"] is True and models[1]["problem"] is None


# --- try one input ---

async def test_try_runs_a_typed_input_and_saves_nothing(client, workspace):
    body = {"experiment": EXP, "vars": {"text": "typed by hand"}, "expected": "positive", "tags": ["mine"]}
    data = (await client.post("/api/try", json=body)).json()

    rows = {row["prompt_id"]: row for row in data["rows"]}
    assert set(rows) == {"p1", "p2"} and all(row["case_id"] == "typed-input" for row in data["rows"])
    assert rows["p1"]["output"] == "It is positive." and not rows["p1"]["passed"]
    assert rows["p1"]["scores"][0]["reason"] == "expected 'positive', got 'It is positive.'"
    assert rows["p2"]["output"] == "positive" and rows["p2"]["passed"]
    assert not (workspace / "runs").exists()


async def test_editing_a_prompt_in_the_browser_changes_the_result(client):
    edited = [{"id": "p1", "template": "Answer in one word. Classify: {text}"},
              {"id": "brand_new", "template": "{text}"}]
    body = {"experiment": EXP, "prompts": edited, "vars": {"text": "x"}, "expected": "neutral"}
    rows = {row["prompt_id"]: row for row in (await client.post("/api/try", json=body)).json()["rows"]}
    assert rows["p1"]["passed"]                 # the edit added the instruction the mock rule looks for
    assert not rows["brand_new"]["passed"]      # a new variant runs too


@pytest.mark.parametrize("change, message", [
    ({"prompts": [{"id": "p1", "template": "Classify {text} for {customer}"}]}, "uses {customer}"),
    ({"prompts": [{"id": "a", "template": "{text}"}, {"id": "a", "template": "{text}!"}]}, "duplicate prompt id"),
    ({"prompts": []}, "prompts"),
    ({"models": ["ghost"]}, "unknown model id: ghost"),
    ({"experiment": "nope.yaml"}, "experiment not found"),
])
async def test_try_reports_problems_as_readable_errors(client, change, message):
    body = {"experiment": EXP, "vars": {"text": "x"}, "expected": "positive", **change}
    response = await client.post("/api/try", json=body)
    assert response.status_code in (400, 404)
    assert message in response.json()["detail"]


async def test_try_without_expected_answer_explains_the_mock(client):
    body = {"experiment": EXP, "vars": {"text": "x"}, "expected": None}
    rows = (await client.post("/api/try", json=body)).json()["rows"]
    assert rows[0]["status"] == "error" and "needs the test case's expected answer" in rows[0]["error"]["message"]


# --- run the whole test set ---

async def test_plan_then_run_then_everything_is_cached(client, workspace):
    assert (await client.post("/api/plan", json={"experiment": EXP})).json() == {
        "total": 6, "cached": 0, "to_call": 6, "prompts": 2, "models": 1, "cases": 3}

    job = await run_all(client, name="first")
    assert job == {"id": job["id"], "status": "done", "done": 6, "total": 6, "cached": 0, "run_id": "first", "error": None}
    assert (workspace / "runs" / "first" / "results.jsonl").is_file()
    assert (await client.post("/api/plan", json={"experiment": EXP})).json()["cached"] == 6

    summary = (await client.get("/api/runs/first")).json()
    assert summary["totals"]["cases"] == 6 and summary["totals"]["passed"] == 3
    assert [(g["prompt_id"], g["passed"]) for g in summary["groups"]] == [("p1", 0), ("p2", 3)]


async def test_run_saves_the_prompts_edited_in_the_browser(client, workspace):
    edited = [{"id": "p1", "template": "Answer in one word. Classify: {text}"}]
    job = await run_all(client, prompts=edited, name="edited")
    snapshot = yaml.safe_load((workspace / "runs" / "edited" / "experiment.yaml").read_text(encoding="utf-8"))
    assert snapshot["prompts"] == [{"id": "p1", "template": "Answer in one word. Classify: {text}"}]
    assert (await client.get(f"/api/runs/{job['run_id']}")).json()["totals"]["passed"] == 3
    # the experiment file itself is untouched
    assert "Answer in one word" not in (workspace / EXP).read_text(encoding="utf-8")


async def test_unnamed_run_gets_a_timestamped_id_and_runs_are_listed_newest_first(client):
    first = await run_all(client, name="older")
    second = await run_all(client)
    assert first["run_id"] == "older" and "-test-exp-" in second["run_id"]
    listed = (await client.get("/api/runs")).json()
    assert {run["run_id"] for run in listed} == {"older", second["run_id"]}
    assert listed[0]["cells"] == 6 and listed[0]["mock"] is True and listed[0]["pass_rate"] == 0.5


async def test_ticking_every_model_is_not_a_partial_run(write_experiment, tmp_path):
    write_experiment({"models": [MOCK, {"id": "m2", "provider": "mock"}]})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(tmp_path)), base_url=BASE) as http:
        await run_all(http, name="all", models=["m1", "m2"])
        await run_all(http, name="one", models=["m2"])
        runs = {run["run_id"]: run for run in (await http.get("/api/runs")).json()}
        assert runs["all"]["partial"] is False and runs["all"]["cells"] == 12
        assert runs["one"]["partial"] is True and runs["one"]["cells"] == 6


@pytest.mark.parametrize("name", ["../escape", "a/b", "with space"])
async def test_bad_run_name_is_rejected_before_anything_runs(client, workspace, name):
    response = await client.post("/api/runs", json={"experiment": EXP, "name": name})
    assert response.status_code == 400 and "invalid run name" in response.json()["detail"]
    assert not (workspace / "runs").exists()


async def test_run_that_cannot_start_reports_why(client):
    response = await client.post("/api/runs", json={"experiment": EXP, "models": ["ghost"]})
    assert response.status_code == 400 and "unknown model id" in response.json()["detail"]
    assert (await client.get("/api/jobs/nope")).status_code == 404


# --- report and compare ---

async def test_report_is_served_with_a_link_back_to_the_app(client):
    await run_all(client, name="r1")
    page = (await client.get("/runs/r1/report")).text
    assert "prompt-lab report" in page and "Back to prompt-lab" in page and 'href="/#runs"' in page
    assert "Side by side" in page


async def test_compare_two_runs_made_in_the_app(client):
    await run_all(client, name="before")
    edited = [{"id": "p1", "template": "Answer in one word. Classify: {text}"},
              {"id": "p2", "template": "Classify quickly: {text}"}]
    await run_all(client, prompts=edited, name="after")

    result = (await client.get("/api/compare", params={"a": "before", "b": "after"})).json()
    assert result["changed_prompts"] == ["p1", "p2"]
    assert result["counts"]["fixes"] == 3 and result["counts"]["regressions"] == 3
    assert {r["prompt_id"] for r in result["regressions"]} == {"p2"}
    assert {f["prompt_id"] for f in result["fixes"]} == {"p1"}


@pytest.mark.parametrize("path", ["/api/runs/nope", "/runs/nope/report", "/api/compare?a=nope&b=nope",
                                  "/api/runs/..", "/runs/%2e%2e/report"])
async def test_unknown_or_unsafe_run_is_404(client, path):
    assert (await client.get(path)).status_code == 404


# --- it only answers its own pages ---

async def test_experiment_path_cannot_leave_the_workspace(client, workspace):
    outside = workspace.parent / "outside.yaml"
    outside.write_text((workspace / EXP).read_text(encoding="utf-8"), encoding="utf-8")
    for experiment_id in ("../outside.yaml", str(outside), "cases.jsonl"):
        assert (await client.get("/api/experiment", params={"id": experiment_id})).status_code == 404


async def test_requests_from_another_site_are_refused(client, workspace):
    body = {"experiment": EXP}
    foreign = await client.post("/api/runs", json=body, headers={"origin": "https://evil.example"})
    assert foreign.status_code == 403 and not (workspace / "runs").exists()
    own = await client.post("/api/plan", json=body, headers={"origin": BASE})
    assert own.status_code == 200
    localhost = await client.post("/api/plan", json=body, headers={"origin": "http://localhost:8321"})
    assert localhost.status_code == 200


async def test_wrong_host_header_is_refused(client):
    response = await client.get("/api/status", headers={"host": "attacker.example"})
    assert response.status_code == 403
