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
    if not _authorized(req):
        print("  submit -> 401 (bad key)")
        return web.json_response({"error": "unauthorized"}, status=401)

    body = await req.json()

    # Mirror the real validator: schema is checked BEFORE participant matching,
    # and a missing or empty transcript 400s without ever reaching the lookup.
    if not (body.get("transcript") or "").strip():
        print("  submit -> 400 (no transcript text)")
        return web.json_response(
            {"error": "Malformed payload: Must provide raw transcript text or audio segments"},
            status=400,
        )
    if not isinstance(body.get("callDurationSec"), (int, float)):
        print("  submit -> 400 (no callDurationSec)")
        return web.json_response({"error": "Malformed payload: callDurationSec"}, status=400)

    if body.get("phoneNumber") not in ROSTER:
        print(f"  submit {body.get('phoneNumber')} -> 404 (not on the roster)")
        return web.json_response({"error": "no match"}, status=404)

    print("\n" + "=" * 72)
    print(f"TRANSCRIPT ACCEPTED  {body['phoneNumber']}  {body['callDurationSec']}s")
    if "callStatus" in body:
        print(f"callStatus: {body['callStatus']}")
    print("=" * 72)
    print(body["transcript"])
    print("=" * 72 + "\n")
    return web.json_response({"ok": True}, status=202)


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
