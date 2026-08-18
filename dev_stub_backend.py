"""A stand-in for the Cardinal backend, so the identified-caller path can be
tested before the real resolve endpoint exists.

    .venv/bin/python dev_stub_backend.py

Then, in another shell:

    CARDINAL_API_BASE=http://127.0.0.1:8899 \
    MBA606_FAKE_CALLER=+15025551234 \
    lk agent console test_agent.py

It implements both endpoints with the contract proposed in
BACKEND_INTEGRATION.md and prints whatever it receives, so you can read the
transcript the agent actually submitted at the end of a call.

Development only -- no auth beyond a shared key, no persistence, no TLS.
"""

from __future__ import annotations

import argparse
import json
from aiohttp import web

# Anything not in here 404s, which is how you exercise the unidentified path.
ROSTER = {
    "+15025551234": {"participantId": "P_TEST_1", "displayName": "Sarah", "consentOnFile": False},
    "+15025554321": {"participantId": "P_TEST_2", "displayName": "Ken", "consentOnFile": True},
}

API_KEY = "cardinal-voice-agent-secret-key-2026"

# Everything the stub accepted, for inspection at the end of a test run.
received: list[dict] = []


def _authorized(req: web.Request) -> bool:
    if req.headers.get("x-api-key") == API_KEY:
        return True
    return req.headers.get("Authorization") == f"Bearer {API_KEY}"


async def resolve(req: web.Request) -> web.Response:
    if not _authorized(req):
        print("  resolve -> 401 (bad key)")
        return web.json_response({"error": "unauthorized"}, status=401)

    phone = (await req.json()).get("phoneNumber")
    student = ROSTER.get(phone)

    if student is None:
        print(f"  resolve {phone} -> 404 (not on the roster)")
        return web.json_response({"error": "no match"}, status=404)

    print(f"  resolve {phone} -> 200 {student['displayName']}")
    return web.json_response(student)


async def submit(req: web.Request) -> web.Response:
    """Mirrors the Option A consent contract, in their stated validation order."""
    if not _authorized(req):
        print("  submit -> 401 (bad key)")
        return web.json_response({"error": "unauthorized"}, status=401)

    body = await req.json()
    consent = body.get("recordingConsent")
    transcript = body.get("transcript")

    # Schema before matching. Consent is checked first and fails closed.
    if not isinstance(consent, bool):
        print("  submit -> 400 (recordingConsent missing)")
        return web.json_response({"error": "recordingConsent is required"}, status=400)
    if not isinstance(body.get("callDurationSec"), (int, float)):
        print("  submit -> 400 (callDurationSec missing)")
        return web.json_response({"error": "Malformed payload: callDurationSec"}, status=400)
    if consent and not (transcript or "").strip():
        print("  submit -> 400 (consent true, no transcript)")
        return web.json_response(
            {"error": "Malformed payload: Must provide raw transcript text or audio segments"},
            status=400,
        )
    if not consent and transcript is not None:
        print("  submit -> 400 (consent false, transcript present)")
        return web.json_response(
            {"error": "transcript must be absent when recordingConsent is false"}, status=400
        )

    phone = body.get("phoneNumber")
    matched = phone in ROSTER

    if not consent:
        # Stored either way -- the audit trail survives an unmatched number.
        print(f"  submit {phone} -> 202 recorded_declined (matched={matched})")
        received.append(body)
        return web.json_response({"status": "recorded_declined"}, status=202)

    if not matched:
        print(f"  submit {phone} -> 404 (not on the roster)")
        return web.json_response({"error": "no match"}, status=404)

    print("\n" + "=" * 72)
    print(f"ACCEPTED  {phone}  {body['callDurationSec']}s  consentAt={body.get('consentCapturedAt')}")
    print("=" * 72)
    print(transcript)
    print("=" * 72 + "\n")
    received.append(body)
    return web.json_response({"status": "accepted"}, status=202)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()

    app = web.Application()
    # Kept only so a stray call is visibly 404-shaped rather than a connection
    # error; the real backend has no such route and the agent no longer calls it.
    app.router.add_post("/api/webhooks/phone-agent/resolve", resolve)
    app.router.add_post("/api/webhooks/phone-agent", submit)

    print(f"stub backend on http://127.0.0.1:{args.port}")
    print("roster:", ", ".join(f"{k} -> {v['displayName']}" for k, v in ROSTER.items()))
    print("any other number 404s, which exercises the unidentified path\n")
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
