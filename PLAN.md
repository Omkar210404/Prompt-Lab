# PLAN — prompt-lab

Status: **approved** (2026-10-11). Decisions D1, D2, D4 and D5 accepted as proposed; D3 (a real-model run) is left for after the build. The README is the up-to-date description; this file records the original plan.

prompt-lab runs a matrix of prompt variants × models × test cases, scores every output with pluggable scorers, saves the run to disk, and shows the results side by side. The point is to measure prompt quality instead of eyeballing a few outputs.

## 1. Environment (inspected 2026-10-11)

| Item | Finding | Consequence |
|---|---|---|
| Python | 3.11.9 (Microsoft Store build) | Target 3.11+. Use a project `.venv` with `python -m venv` / `python -m pip`. |
| Global packages | httpx 0.28.1, PyYAML 6.0.3, Jinja2 3.1.6, **pydantic 1.10**; no typer, rich, jsonschema, pytest | Ignore globals. Install current versions into `.venv` and pin `requirements.txt` from what actually installs. |
| git | 2.49, global identity `Omkar210404` | `git init -b main` inside `prompt-lab/`. |
| GitHub CLI | not installed | I can't create the GitHub repo or push. You push; the badge URL assumes `Omkar210404/prompt-lab` (D2). |
| Internet | PyPI reachable | Dependencies can be installed. Tests and demo still run offline. |
| LLM keys | none in the environment | Everything is built and verified on the mock provider. The OpenAI-compatible provider is tested against `httpx.MockTransport` only, unless you give a key (D3). |
| Browser | Chrome and Edge installed | The README screenshot is taken with headless Chrome from the real mock report. No Playwright needed. |

## 2. Experiment file format

One YAML file per experiment. Paths are relative to the YAML file.

```yaml
name: support-email-extraction
description: Extract ticket fields from a short support email as JSON.
dataset: datasets/support_emails.jsonl

prompts:
  - id: v1_bare
    template: |
      Extract the order id, category, refund amount and sentiment from this email.

      {email}
  - id: v2_json_rules
    system: You are a support-ticket parser.        # optional
    template: |
      ...respond with only a JSON object with keys order_id, category, ...
      {email}

models:
  - id: mock-careful
    provider: mock
    model: mock-careful
    mock: { rules: [...] }                # see section 5
  - id: groq-llama                        # example of a real model, not in the demo
    provider: openai_compatible
    model: <model name>
    base_url: https://api.groq.com/openai/v1
    api_key_env: GROQ_API_KEY             # the NAME of the env var, never the key
    params: { temperature: 0, max_tokens: 300 }

scorers:                                  # applied to every case
  - type: json_valid
  - type: json_schema
    schema_file: schemas/ticket.schema.json
  - type: json_fields
    fields: [order_id, category, sentiment]
  - type: numeric
    field: refund_amount
    abs_tol: 0.01

run:
  concurrency: 4
  timeout_seconds: 30
  max_retries: 2
  cache: true

prices_file: prices.yaml                  # optional, user-editable
```

Test set, one JSON object per line:

```json
{"id": "email-007", "vars": {"email": "Hi, order ORD-48213 arrived broken..."}, "expected": {"order_id": "ORD-48213", "category": "refund", "refund_amount": 49.99, "sentiment": "negative"}, "checks": [{"type": "not_contains", "value": "Sure"}], "tags": ["amount_with_currency"]}
```

- `expected` is a string or an object. `checks` are extra scorers for that one case, on top of the experiment-level `scorers`.
- **Template variables:** only `{name}` where `name` is a simple identifier is substituted, so JSON examples inside few-shot prompts (`{"order_id": ...}`) are left alone. `{{name}}` writes a literal. A `{name}` that no test case provides is a load-time error, not a silent blank.
- **Validation at load** (Pydantic v2): duplicate ids, unknown scorer or provider types, missing variables, bad JSONL lines (reported with line number), missing schema files.
- **Config hash:** SHA-256 over the resolved experiment (prompts, models, params, scorers) plus the dataset file hash and any schema files. API keys are never part of it.

Price table (`prices.yaml`), USD per 1M tokens, keyed by model name:

```yaml
mock-careful: { input: 0.10, output: 0.40, note: "Fictional price so the demo can show the cost column." }
# your-real-model: { input: ?, output: ?, source: "<pricing page URL>", checked: "<date>" }
```

The repo ships **no real provider prices**. A model with no entry, or a provider that reports no token usage, gets cost `n/a`, never `0`.

## 3. Scorers

Each scorer returns `ScoreResult(passed: bool, score: 0..1, reason: str)`. Registered by name with a decorator, so adding one is a new class plus one line. A case passes when all its scorers pass; its score is the mean. A scorer can be marked `required: false` to make it advisory.

| Type | What it checks | Main options |
|---|---|---|
| `exact_match` | output equals expected | `ignore_case`, `strip`, `normalize_whitespace`, `expected_field` |
| `contains` | substring(s) present | `value` / `values`, `mode: all\|any`, `ignore_case` |
| `not_contains` | substring(s) absent | `value` / `values`, `ignore_case` |
| `regex` | pattern matches | `pattern`, `mode: search\|fullmatch`, `ignore_case` |
| `json_valid` | whole output parses as JSON (strict by default: prose around JSON fails) | `allow_fenced` |
| `json_schema` | parsed output validates against a JSON Schema (`jsonschema` library) | `schema` inline or `schema_file` |
| `json_fields` | listed fields of the parsed output equal `expected`; score is the fraction matching | `fields`, `ignore_case` |
| `numeric` | number within tolerance of expected | `abs_tol`, `rel_tol`, `field` |
| `length` | length limits | `min_chars`, `max_chars`, `min_words`, `max_words` |
| `llm_judge` | a judge model applies a written rubric | `rubric`, `judge` (model id), `scale: binary\|1-5`, `threshold` |

`json_fields` is one addition to your list: without it the extraction demo could only use whole-string exact match, which fails on key order and whitespace. `exact_match`, `contains`, `regex` and `numeric` also accept `field:` to run on one field of a JSON output.

**LLM-as-judge**, built to be used carefully:
- The judge must reply with JSON, reasoning first and verdict second. Default scale is binary pass/fail; temperature 0.
- Judge output that isn't valid is recorded as a **judge error**, shown separately. It is never silently counted as a pass or a fail.
- A warning is printed when the judge is the same model as the one being tested.
- Judge calls go through the same cache, and judge tokens and cost are reported separately from the models under test.
- The report shows the judge's reasoning for every case, so a person can audit it.
- README section on limits: position and verbosity bias, self-preference, run-to-run variance, rubric sensitivity, and no ground truth. Advice: use deterministic scorers first, keep rubrics binary and specific, use a different and stronger judge model, hand-label a sample and check agreement before trusting it.
- Tested offline with the mock provider as the judge. The demo's headline numbers use deterministic scorers only.

## 4. Runner, cache, cost

- **Matrix:** every prompt × model × test case is one cell. `--prompt`, `--model` and `--limit` narrow it. `--dry-run` prints how many calls would be made and how many are already cached, before any paid call.
- **Concurrency:** asyncio with a global semaphore (`run.concurrency`) and an optional per-model limit for strict free tiers.
- **Timeouts and retries:** per-call timeout. Timeouts, HTTP 429 and 5xx are retried with exponential backoff, honouring `Retry-After`. 401/403 and other 4xx are not retried. The retry wrapper and error hierarchy are adapted from `multi-ai-agent-system/app/llm/` (copied, not imported).
- **A failed cell never aborts the run.** It is saved with `status: error` and the error type, and counts as a failure.
- **Cache:** on disk in `.promptlab/cache/`. Key is the SHA-256 of provider type, base URL, model, system prompt, rendered prompt and params. Only successful responses are cached. Scoring is never cached, so changing scorers and re-running costs nothing. `--no-cache` skips it; `--refresh` overwrites it.
- **Metrics per cell:** latency (measured), `prompt_tokens` / `completion_tokens` only when the provider reports them, cost from the price table. Cache hits are flagged and keep the latency of the original call. The summary separates total cost from cost actually spent in this run.

## 5. Providers

- `Provider` ABC with one async `complete(request) -> response` method, plus a small registry (`provider: mock | openai_compatible`).
- **`OpenAICompatibleProvider`:** plain `httpx` POST to `{base_url}/chat/completions`. The key is read from the env var named in `api_key_env`; a missing key is a clear config error before any call is made. `.env` in the working directory is loaded for convenience.
- **`MockProvider`:** deterministic and offline. It does not generate text. It starts from the test case's expected answer and applies **scripted rules written in the experiment YAML**:
  - Conditions: text the prompt contains or lacks (regex), case tags or ids, input regex, and a deterministic `rate` (stable hash of the case id, not random).
  - Behaviours: `correct`, `wrong` (given value), `invalid_json`, `prose_wrapped`, `fenced`, `amount_as_string`, `wrong_field`, `empty`, `error`, `timeout`, plus `latency_ms`.
  - It reports synthetic token counts (word-based) so the cost path is exercised.

  This is how three prompt variants produce a different, explainable result: for example, a rule says "if the prompt does not ask for JSON only, wrap the answer in prose". **The demo differences are therefore authored by those rules, not discovered.** They show how the tool works, not how any model behaves. Every surface says so: `summary.json` carries a `mock: true` flag, and the CLI table, the HTML report and the README show a "mock results, not a benchmark" banner.

## 6. Run and storage format

```
runs/<run_id>/            run_id = <UTC timestamp>-<experiment-slug>-<config hash, 8 chars>
├── results.jsonl         one line per cell
├── summary.json
├── experiment.snapshot.yaml   resolved config as it was run
└── report.html           written by `report --html`
```

`results.jsonl` row:

```json
{"case_id": "email-007", "prompt_id": "v2_json_rules", "model_id": "mock-careful", "status": "ok", "output": "{...}", "error": null, "latency_ms": 12.4, "cached": false, "attempts": 1, "usage": {"prompt_tokens": 84, "completion_tokens": 31}, "cost_usd": 0.0000208, "passed": false, "score": 0.75, "scores": [{"scorer": "json_valid", "passed": true, "score": 1.0, "reason": "valid JSON object"}, {"scorer": "numeric", "passed": false, "score": 0.0, "reason": "refund_amount: expected 49.99, got \"$49.99\" (not a number)"}]}
```

`summary.json`: run id, experiment name, config hash, dataset hash and size, start and end timestamps (UTC), tool and Python version, `mock` flag, cache hits and misses, a snapshot of the prices used, and per prompt × model: cases, passed, pass rate, mean score, errors, p50 and p95 latency, tokens, cost. Also a pass rate per scorer.

`.gitignore` ignores `runs/*` except the two committed demo runs, `runs/demo/` and `runs/demo-shortened/`. `run --name demo` gives a run a fixed folder name.

## 7. Reports and compare

**CLI table** (`report RUN`): one row per prompt × model with pass rate, passed/total, mean score, errors, p50 latency, tokens and cost. `RUN` accepts a run id, a unique prefix, or `latest`.

**HTML report** (`report RUN --html [--open]`): one self-contained file. CSS, JS, data and the SVG chart are inline, with no CDN, so it opens offline. Dark theme.
1. Header: experiment, run id, timestamp, config hash, dataset size, mock banner when relevant.
2. Summary table per prompt × model (same numbers as the CLI table).
3. Pass-rate chart: grouped horizontal bars, one group per prompt, one bar per model.
4. Side-by-side grid: rows are test cases, columns are prompt variants, with a model switcher. Each cell shows pass/fail and score. Filters: failed only, by tag.
5. Drill-down: clicking a case shows its input, expected output, and each variant's output next to each other with every scorer's reason.
6. Worst failures: cases ranked by lowest score and by how many variants fail them.
7. Per-scorer pass rates, and the prompt templates used.

**Compare** (`compare RUN_A RUN_B`): cells are joined on case, prompt and model.
- Lists regressions (passed in A, fail in B) first, then fixes, then pass-rate and score changes per prompt × model, then cells only in one run.
- Warns when the dataset or config hash differs and says which.
- `--fail-on-regression` exits with code 1 for CI use; `--json` for machine output.

## 8. CLI

`prompt-lab` (Typer), also `python -m promptlab`:

| Command | Does |
|---|---|
| `init [DIR]` | Writes a starter experiment, dataset, `prices.yaml` and `.env.example` |
| `run EXPERIMENT.yaml` | Runs the matrix, saves the run, prints the table. Flags: `--name --concurrency --no-cache --refresh --limit --prompt --model --dry-run` |
| `report RUN` | CLI table; `--html`, `--open`, `--out` |
| `compare RUN_A RUN_B` | Diff with regressions; `--fail-on-regression`, `--json` |
| `list` | Saved runs with date, experiment and pass rates; `--scorers` lists registered scorers |

## 9. Sample experiment (synthetic, mock)

- **Dataset:** `examples/datasets/support_emails.jsonl`, 24 short support emails that I write by hand. Labelled synthetic in the file header comment, the README and the report. Each has `order_id`, `category` (refund, shipping, billing, technical, account), `refund_amount` and `sentiment`. Tags mark the hard cases: no order id, amount with a currency symbol, two order ids, mixed sentiment.
- **`examples/support_extraction.yaml`:** three prompts of increasing quality (`v1_bare`, `v2_json_rules` with keys, allowed values and "JSON only", `v3_fewshot` adding number formatting rules and two worked examples), run on two mock models with different rule sets.
- **`examples/sentiment.yaml`:** the same emails, plain-label output, scored with `exact_match`, `regex` and `length`.
- **`examples/support_extraction_shortened.yaml`:** `v3_fewshot` with the examples removed, so `compare` has a real regression to show between the two committed demo runs.
- README numbers are copied from these runs after they execute. I'm not predicting any pass rates here.

## 10. File layout

```
prompt-lab/
├── promptlab/
│   ├── __init__.py  __main__.py
│   ├── cli.py                  # Typer app: init, run, report, compare, list
│   ├── config.py               # Pydantic models, YAML/JSONL loading, templating, config hash
│   ├── errors.py               # ConfigError, ProviderError hierarchy (retryable flag)
│   ├── providers/
│   │   ├── base.py             # Provider ABC, CompletionRequest/Response
│   │   ├── mock.py             # scripted rules
│   │   ├── openai_compatible.py
│   │   └── registry.py         # build_provider(), retry wrapper
│   ├── scorers/
│   │   ├── base.py             # Scorer ABC, ScoreResult, @register
│   │   ├── text.py             # exact_match, contains, not_contains, regex, length
│   │   ├── json_scorers.py     # json_valid, json_schema, json_fields
│   │   ├── numeric.py
│   │   └── llm_judge.py
│   ├── runner.py               # matrix, concurrency, timeout, scoring
│   ├── cache.py
│   ├── pricing.py
│   ├── storage.py              # run dirs, results.jsonl, summary.json, run lookup
│   ├── compare.py
│   ├── report/
│   │   ├── table.py            # rich table
│   │   ├── html.py
│   │   └── templates/report.html.j2
│   └── scaffold/               # files copied by `init`
├── examples/                   # experiments, datasets/, schemas/, prices.yaml
├── runs/demo/  runs/demo-shortened/
├── tests/
├── docs/screenshots/
├── .github/workflows/ci.yml
├── .claude/settings.json       # attribution off; .claude/ is gitignored
├── .env.example  .gitignore  LICENSE  README.md  PLAN.md
├── pyproject.toml              # package metadata, `prompt-lab` entry point, pytest config
└── requirements.txt  requirements-dev.txt     # pinned
```

Dependencies: typer, rich, pyyaml, pydantic v2, httpx, jsonschema, jinja2, python-dotenv. Dev: pytest, pytest-asyncio.

## 11. Tests (all offline)

A `conftest.py` fixture blocks real socket connections, so a test that tried to reach the internet would fail.

| File | Covers |
|---|---|
| `test_scorers_text.py` | exact_match, contains, not_contains, regex, length: pass, fail and option cases |
| `test_scorers_json.py` | json_valid (strict, fenced), json_schema (violations listed), json_fields (partial score) |
| `test_scorers_numeric.py` | abs and rel tolerance, non-numeric output, `field` |
| `test_llm_judge.py` | pass, fail, malformed judge reply, judge call error, same-model warning |
| `test_config.py` | valid load, missing variable, duplicate ids, unknown scorer, bad JSONL line, braces in templates, config hash stable and sensitive to changes |
| `test_mock_provider.py` | each behaviour, rule order, determinism of `rate` |
| `test_openai_compatible.py` | via `httpx.MockTransport`: success with usage, 401, 429 with Retry-After, 500, timeout, malformed body, missing key |
| `test_runner.py` | matrix size, concurrency limit respected, timeout becomes an error row, retry then success, no retry on 401, one failing cell doesn't stop the run |
| `test_cache.py` | second run makes zero provider calls, key changes with prompt/model/params, errors not cached, `--no-cache`, `--refresh` |
| `test_pricing.py` | cost maths, missing price or usage gives `None`, cached calls excluded from spend |
| `test_storage.py` | write and read round trip, run lookup by id, prefix and `latest` |
| `test_compare.py` | regressions, fixes, added and removed cells, hash mismatch warning, exit code |
| `test_report.py` | HTML contains the data and the mock banner, no external URLs, table renders |
| `test_cli.py` | `init` → `run` → `report --html` → `compare` → `list` end to end in a temp dir |

CI: GitHub Actions on every push and pull request, Python 3.11 on Ubuntu. It runs `pytest`, then the mock demo end to end as a smoke test.

## 12. Build order (commit after each step, tests passing)

1. Scaffold: `.gitignore`, `.claude/settings.json`, LICENSE, pyproject, venv, pinned requirements
2. Config and templating, with tests
3. Providers (base, mock, OpenAI-compatible, retry), with tests
4. Scorers, with tests
5. Cache, pricing, runner, with tests
6. Storage and CLI table; `run`, `list`, `report`
7. HTML report
8. Compare
9. `init` scaffold and the CLI end-to-end test
10. Sample experiments and dataset; generate the demo runs
11. README (Mermaid diagram, real command output, real screenshot), CI workflow and badge
12. Definition of Done checklist from `brain.md`; final report

## 13. Risks

| Risk | Mitigation |
|---|---|
| Mock demo numbers read as real model results | Mock banner in the CLI, HTML and README; `mock: true` in `summary.json`; the rules that produce the differences are in the YAML and explained in the README. |
| The real provider path is never run against a live API | Tested with `httpx.MockTransport` for every status path. The README says it was not run against a live provider, unless you supply a key (D3). |
| Braces in prompts (JSON examples) break templating | Identifier-only substitution, a `{{ }}` escape, load-time errors for unknown variables, tested. |
| Cache returns stale output after a silent provider-side model change, or freezes one sample at temperature above 0 | The key covers everything prompt-lab sends; `--refresh` and `--no-cache`; documented in limitations. |
| Free-tier rate limits | Concurrency limit, per-model limit, backoff with `Retry-After`, `--limit` and `--dry-run`. |
| Cost figures taken as exact | Labelled "estimated"; computed only from reported tokens and your price table; `n/a` when either is missing; no real prices shipped. |
| LLM judge trusted too much | Judge errors kept separate, reasoning shown, same-model warning, README limits section. Not used for the demo's headline numbers. |
| Windows console can't print rich's box characters | Force UTF-8 output and fall back to ASCII boxes; README output is pasted from real runs. |
| HTML report grows with run size (all rows are embedded) | Fine for hundreds to low thousands of cells; stated in limitations. |
| Scope | The build order is incremental; each step leaves a working, tested tool. |

## 14. Decisions and assumptions to confirm

- **D1, UI.** The self-contained HTML report (dark theme, opened with `report --open`) is the app screen for this project. No web server or Streamlit app. Say so if you want a served UI as well.
- **D2, GitHub.** The CI badge will point at `Omkar210404/prompt-lab`. I commit locally only; you create the repo and push.
- **D3, real model run.** Optional. If you put a Groq (or similar) key in `prompt-lab/.env` after the build, I can do one small real run and add a dated results table with model names and test set size, labelled as a small test. Otherwise the README shows mock results only.
- **D4, extra scorer.** `json_fields` is added beyond your list (reason in section 3).
- **D5, two committed demo runs** (`runs/demo/`, `runs/demo-shortened/`) so the README can show real `compare` output.
