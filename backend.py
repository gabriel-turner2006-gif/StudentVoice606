"""Client for the Cardinal Intelligence phone-agent webhook.

One call matters: submit_call(), once, when the call ends. The backend is
an end-of-call ingestion webhook rather than a lifecycle API -- caller matching
happens there, on the phone number, after the conversation is over.

There is deliberately no caller-lookup call. The backend exposes no
side-effect-free way to resolve a phone number: the only path to its matching
logic is posting a complete call, which stores a record and queues extraction,
so using it as a lookup would manufacture junk. Identity is therefore resolved
by them, after the fact, from the number we post.

Everything here degrades rather than raising. A backend outage should cost us the
student's record, never the conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger("mba606.backend")

BASE_DIR = Path(__file__).parent

# Where a POST goes to die. Anything in here is a real conversation that a
# consenting student expected to land on their record, so it is appended rather
# than overwritten and is meant to be replayed by hand.
DEAD_LETTER_PATH = BASE_DIR / "logs" / "undelivered_calls.jsonl"

SUBMIT_PATH = "/api/webhooks/phone-agent"

# The submit runs during worker shutdown, which gives us room to retry.
SUBMIT_TIMEOUT_S = 15.0
SUBMIT_ATTEMPTS = 3


class CardinalClient:
    """Talks to the phone-agent webhook. Never raises at the call sites."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        base = base_url if base_url is not None else os.getenv("CARDINAL_API_BASE", "")
        self._base_url = base.rstrip("/")
        self._api_key = api_key if api_key is not None else os.getenv("CARDINAL_API_KEY", "")
        self._session: aiohttp.ClientSession | None = None

        if not self.enabled:
            logger.warning(
                "CARDINAL_API_BASE is not set -- running with no backend. Every call "
                "will be anonymous and no transcript will be submitted."
            )
        elif not self._api_key:
            logger.warning("CARDINAL_API_BASE is set but CARDINAL_API_KEY is empty; expect 401s")

    @property
    def enabled(self) -> bool:
        return bool(self._base_url)

    def _client(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"x-api-key": self._api_key, "content-type": "application/json"}
            )
        return self._session

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def submit_call(
        self,
        *,
        phone_number: str,
        recording_consent: bool,
        call_duration_sec: int,
        transcript: str | None = None,
        consent_captured_at: str | None = None,
        call_status: str | None = None,
    ) -> bool:
        """POST a finished call. Returns whether the backend accepted it.

        The consent contract is strict on their side and mirrored here so a bug
        fails locally rather than as an opaque 400:

          consent true  -> transcript required, non-empty
          consent false -> transcript must be ABSENT, not a placeholder
          consent missing -> rejected; silence is never consent

        A declined call is still posted. It stores an audit record -- a call
        happened and consent was refused -- and is kept even when the phone
        number matches no participant, which is the one case where a 404 does
        not apply. Nothing is queued for extraction.
        """
        if recording_consent and not (transcript or "").strip():
            logger.error("refusing to post: consent given but transcript is empty")
            return False
        if not recording_consent and transcript is not None:
            logger.error("refusing to post: consent declined but a transcript was passed")
            return False

        payload: dict[str, Any] = {
            "phoneNumber": phone_number,
            "callDurationSec": call_duration_sec,
            "recordingConsent": recording_consent,
        }
        if transcript is not None:
            payload["transcript"] = transcript
        if consent_captured_at is not None:
            payload["consentCapturedAt"] = consent_captured_at
        if call_status is not None:
            payload["callStatus"] = call_status

        what = "transcript" if recording_consent else "declined-call record"

        if not self.enabled:
            # Print what would have gone over the wire. Without this a test call
            # against no backend leaves nothing to inspect, which is precisely when
            # you most want to read the transcript.
            logger.info(
                "no backend configured -- would have posted a %s for %s (%ss):\n%s",
                what,
                phone_number,
                call_duration_sec,
                transcript if transcript is not None else "(no transcript, consent declined)",
            )
            return False

        url = f"{self._base_url}{SUBMIT_PATH}"
        last_error = "unknown"

        for attempt in range(1, SUBMIT_ATTEMPTS + 1):
            try:
                async with self._client().post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=SUBMIT_TIMEOUT_S),
                ) as resp:
                    if resp.status == 202:
                        status = ""
                        try:
                            status = (await resp.json()).get("status", "")
                        except Exception:
                            pass
                        logger.info(
                            "%s accepted for %s (%ss)%s",
                            what,
                            phone_number,
                            call_duration_sec,
                            f" -- {status}" if status else "",
                        )
                        return True

                    body = (await resp.text())[:400]
                    last_error = f"http {resp.status}: {body}"

                    if 400 <= resp.status < 500:
                        # Terminal. 401 is a bad key, 404 is an unmatched number,
                        # 400 is a payload we built wrong -- none improve on retry.
                        logger.error("call rejected, not retrying -- %s", last_error)
                        _dead_letter(payload, last_error)
                        return False

                    logger.warning("submit attempt %d failed -- %s", attempt, last_error)
            except asyncio.TimeoutError:
                last_error = f"timeout after {SUBMIT_TIMEOUT_S}s"
                logger.warning("submit attempt %d timed out", attempt)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("submit attempt %d errored -- %s", attempt, last_error)

            if attempt < SUBMIT_ATTEMPTS:
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

        logger.error("call undeliverable after %d attempts -- %s", SUBMIT_ATTEMPTS, last_error)
        _dead_letter(payload, last_error)
        return False


def _dead_letter(payload: dict[str, Any], error: str) -> None:
    """Persist a call we could not deliver so it can be replayed by hand."""
    try:
        DEAD_LETTER_PATH.parent.mkdir(exist_ok=True)
        record = {"failed_at": time.time(), "error": error, "payload": payload}
        with DEAD_LETTER_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.error("wrote undelivered call to %s", DEAD_LETTER_PATH)
    except Exception:
        # Last resort: the transcript at least exists in the logs.
        logger.exception("could not write dead-letter record; payload was %r", payload)
