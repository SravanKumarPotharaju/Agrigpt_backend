You are running the AgriGPT assignment evaluation suite.

The student's backend URL is: $ARGUMENTS

If no URL was provided (i.e., $ARGUMENTS is empty or still says "$ARGUMENTS"), stop immediately and tell the user:
  "Usage: /eval-assignment <backend-url>  (e.g. /eval-assignment https://myapp.onrender.com)"

---

## STEP 1 — Reachability + URL Type Detection

Run this bash command. It auto-strips `/docs`, `/redoc`, `/swagger` suffixes, probes the root to detect UI vs FastAPI, and checks `/hi` as optional info:

```bash
python3 -c "
import requests, sys, json, re

raw_url = '$ARGUMENTS'.rstrip('/')
url = re.sub(r'/(docs|redoc|openapi\.json|swagger)$', '', raw_url, flags=re.IGNORECASE)
stripped = (url != raw_url)
result = {'reachable': False, 'submitted_url': raw_url, 'resolved_url': url, 'stripped_suffix': stripped, 'hi_status': None, 'url_type': None, 'error': None, 'note': None}

try:
    r = requests.get(url, timeout=15)
    ct = r.headers.get('content-type', '')
    body = r.text.strip()
    if 'text/html' in ct or body.startswith('<!') or body.lower().startswith('<html'):
        result['url_type'] = 'UI'
    else:
        result['url_type'] = 'FASTAPI'
    result['reachable'] = True
except requests.exceptions.Timeout:
    result['error'] = 'TIMEOUT'
except Exception as e:
    result['error'] = str(e)

if result['reachable'] and result['url_type'] == 'FASTAPI':
    try:
        h = requests.get(f'{url}/hi', timeout=10)
        result['hi_status'] = h.status_code
        result['note'] = '/hi present' if h.status_code == 200 else '/hi missing (not required)'
    except Exception:
        result['note'] = '/hi unreachable (not required)'

print(json.dumps(result, indent=2))
"
```

**Evaluate the output:**

- If `error` is `TIMEOUT` or not null → stop and report:
  `"❌ Server unreachable at $ARGUMENTS — did not respond (timeout or connection error)."`

- If `url_type` is `UI` → stop and report:
  ```
  ❌ Wrong URL submitted.
     $ARGUMENTS returned HTML — this is a frontend/UI, not the FastAPI backend.
     The student must submit their backend API URL instead.
  ```

- If `stripped_suffix` is `true` → note in the report:
  `"ℹ️  /docs suffix stripped — using resolved base URL: <resolved_url>"`

- If `url_type` is `FASTAPI` → use `resolved_url` as BASE_URL in Step 2.

In the report, show `/hi` status as informational only — ✅ if present, ℹ️ if missing.

---

## STEP 2 — Discover Endpoint + Run All Test Cases

Run this single Python script. It auto-discovers the chat endpoint (trying standard paths first, then scanning the OpenAPI spec), then runs all 12 tests **sequentially with smart retry on rate-limit** to avoid Gemini free-tier quota exhaustion:

```bash
python3 -c "
import requests, uuid, json, sys, re, time

raw_url = '$ARGUMENTS'.rstrip('/')
BASE_URL = re.sub(r'/(docs|redoc|openapi\.json|swagger)$', '', raw_url, flags=re.IGNORECASE)

# 1. Try standard candidates first
CANDIDATES = ['/test/chat', '/chat']
ENDPOINT = None
for candidate in CANDIDATES:
    try:
        probe = requests.post(f'{BASE_URL}{candidate}', json={
            'chatId': 'probe', 'phone_number': 'eval-bot', 'message': 'hi'
        }, timeout=60)
        if probe.status_code != 404:
            ENDPOINT = f'{BASE_URL}{candidate}'
            break
    except Exception:
        pass

# 2. Fall back: scan OpenAPI spec for any POST path containing 'chat'
if not ENDPOINT:
    try:
        spec = requests.get(f'{BASE_URL}/openapi.json', timeout=10).json()
        for path, methods in spec.get('paths', {}).items():
            if 'chat' in path.lower() and 'post' in methods:
                probe = requests.post(f'{BASE_URL}{path}', json={
                    'chatId': 'probe', 'phone_number': 'eval-bot', 'message': 'hi'
                }, timeout=15)
                if probe.status_code != 404:
                    ENDPOINT = f'{BASE_URL}{path}'
                    break
    except Exception:
        pass

if not ENDPOINT:
    print(json.dumps({'endpoint_error': f'No chat endpoint found at {BASE_URL} (tried /test/chat, /chat, and OpenAPI spec scan)', 'results': [], 'all_api_key_issue': False, 'any_rate_limit': False}))
    sys.exit(0)

API_KEY_HINTS = ['api key', 'apikey', 'google_api_key', 'invalid api', 'authentication', 'unauthorized', 'api_key']
RATE_LIMIT_HINTS = ['resource_exhausted', 'rate limit', 'quota exceeded', 'too many requests', '429']

test_cases = [
    ('pest',       'What pests affect rice crops?',                             'simulate_pests'),
    ('pest',       'How do I treat wheat rust disease?',                        'simulate_pests'),
    ('scheme',     'What government schemes are available for farmers?',         'government_schemes'),
    ('scheme',     'What is Kisan Credit Card and how to apply?',               'government_schemes'),
    ('irrelevant', 'What is the capital of France?',                            None),
    ('irrelevant', 'What is 2 + 2?',                                            None),
]

def extract_retry_delay(error_text):
    m = re.search(r'retry[^0-9]*(\d+)', error_text, re.IGNORECASE)
    return int(m.group(1)) + 2 if m else 15

def call_once(question):
    resp = requests.post(ENDPOINT, json={
        'chatId': str(uuid.uuid4()),
        'phone_number': 'eval-bot',
        'message': question
    }, timeout=30)
    raw_body = resp.text[:500]
    try:
        data = resp.json()
        sources = data.get('sources', [])
        response_text = data.get('response', '')[:120]
        server_error = data.get('detail') or data.get('error') or data.get('message', '')
    except Exception:
        data = {}
        sources = []
        response_text = ''
        server_error = raw_body
    combined = (raw_body + str(server_error)).lower()
    return resp.status_code, sources, response_text, server_error, combined

results = []
any_rate_limit = False
all_api_key = True

for category, question, expected in test_cases:
    try:
        status, sources, response_text, server_error, combined = call_once(question)

        # On rate-limit, wait and retry once
        if any(h in combined for h in RATE_LIMIT_HINTS):
            wait = extract_retry_delay(combined)
            time.sleep(wait)
            status, sources, response_text, server_error, combined = call_once(question)

        api_key_issue = any(h in combined for h in API_KEY_HINTS)
        rate_limit_issue = any(h in combined for h in RATE_LIMIT_HINTS)
        if not api_key_issue:
            all_api_key = False
        if rate_limit_issue:
            any_rate_limit = True

        results.append({
            'category': category,
            'question': question,
            'expected': expected,
            'sources': sources,
            'response': response_text,
            'http_status': status,
            'server_error': server_error if status >= 400 else None,
            'api_key_issue': api_key_issue,
            'rate_limit_issue': rate_limit_issue,
            'error': None
        })
    except Exception as e:
        all_api_key = False
        results.append({
            'category': category,
            'question': question,
            'expected': expected,
            'sources': [],
            'response': '',
            'http_status': 0,
            'server_error': None,
            'api_key_issue': False,
            'rate_limit_issue': False,
            'error': str(e)
        })

print(json.dumps({'endpoint_used': ENDPOINT, 'results': results, 'all_api_key_issue': all_api_key, 'any_rate_limit': any_rate_limit}, indent=2))
"
```

**After running:**

- If `endpoint_error` is present → stop and report:
  ```
  ❌ No usable chat endpoint found.
     Tried: /test/chat, /chat, and OpenAPI spec scan — nothing responded.
     The student must implement a POST endpoint with 'chat' in the path.
  ```

- Note `endpoint_used` in the report header — ✅ `/test/chat` (standard) or ℹ️ other path (discovered via OpenAPI).

**Check warnings (add PRE-CHECK WARNINGS section if any apply):**

- If `all_api_key_issue` is `true`:
  ```
  ⚠️  API KEY ISSUE — GEMINI_API_KEY missing or invalid on Render.
      Fix: Render Dashboard → Environment → add GEMINI_API_KEY=<key> → Redeploy.
  ```

- If `any_rate_limit` is `true` (and `all_api_key_issue` is false):
  ```
  ⚠️  GEMINI FREE-TIER RATE LIMIT — some calls hit quota even after retry.
      Results below may undercount passes. Re-run when quota resets (wait ~1 min).
  ```

---

## STEP 3 — Grade Each Result

Use **flexible source matching** — students may name their tools with short names (e.g. `pests`, `schemes`) or full names (`simulate_pests`, `government_schemes`). Match any source that contains the keyword:

- A source **matches pest** if it contains `pest` (case-insensitive)
- A source **matches scheme** if it contains `scheme` (case-insensitive)

**Pest test:**
- PASS if any source in `sources` matches pest (contains "pest")
- FAIL if no source matches pest
- ERROR if `error` field is not null

**Scheme test:**
- PASS if any source in `sources` matches scheme (contains "scheme")
- FAIL if no source matches scheme
- ERROR if `error` field is not null

**Irrelevant test:**
- PASS if NO source matches pest AND NO source matches scheme
- REVIEW if any source matches pest or scheme — agent called a farm tool for a non-farm question
- ERROR if `error` field is not null

---

## STEP 4 — Print the Report

```
============================================================
  AgriGPT Assignment Evaluation Report
  URL  : <the submitted URL>
  Date : <today's date>
============================================================

[HEALTH CHECK]
  ✅  Server reachable (FastAPI backend detected)
  ℹ️  /docs suffix stripped — resolved base: <resolved_url>   ← omit if not stripped
  ✅  GET /hi → 200 OK          ← or →   ℹ️  GET /hi → not present (optional)
  ✅  Endpoint: /test/chat       ← or →   ℹ️  Endpoint: <path> (discovered via OpenAPI)

[PRE-CHECK WARNINGS]          ← omit this section if no warnings
  ⚠️  ...

[PEST TOOL TESTS]  (expected source: simulate_pests / pests)
  ✅ PASS   | <question>
  ❌ FAIL   | <question>  |  got sources: <sources list>
  🔴 ERROR  | <question>  |  <error message>

[SCHEME TOOL TESTS]  (expected source: government_schemes / schemes)
  ✅ PASS   | <question>
  ❌ FAIL   | <question>  |  got sources: <sources list>
  🔴 ERROR  | <question>  |  <error message>

[IRRELEVANT QUESTION TESTS]  (expect: no ag-tool called)
  ✅ PASS    | <question>
  ⚠️ REVIEW  | <question>  |  unexpected sources: <sources list>
  🔴 ERROR   | <question>  |  <error message>

============================================================
  SCORE   : <passed> / <total>  (<percentage>%)
  Passed  : <n>
  Failed  : <n>
  Review  : <n>
  Errors  : <n>
============================================================
```

Count REVIEW as not-passed when calculating score.

---

## Adding New Tools in the Future

To test a new tool, add entries to the `test_cases` list in STEP 2:
```python
('weather', 'What is the weather in Hyderabad?', 'weather_tool'),
```
And add a grading rule in STEP 3:
- PASS if any source contains "weather" (case-insensitive)
- FAIL otherwise
