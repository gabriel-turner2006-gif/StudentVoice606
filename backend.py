"""Client for the Cardinal Intelligence phone-agent backend.

Two calls matter to a phone conversation:

  resolve_caller()     at call start -- turn a caller ID into a student record
  submit_transcript()  at hangup     -- hand the conversation back

The transcript webhook is documented and verified working. The resolve endpoint
is a proposal (see BACKEND_INTEGRATION.md); until it ships, resolve_caller
returns None for everyone and every call runs the anonymous path.

Everything here degrades to "anonymous call" rather than raising. A backend
outage should cost us the student's record, never the conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger("mba606.backend")

BASE_DIR = Path(__file__).parent

# Where a POST goes to die. Anything in here is a real conversation that a
# consenting student expected to land on their record, so it is appended rather
# than overwritten and is meant to be replayed by hand.
DEAD_LETTER_PATH = BASE_DIR / "undelivered_calls.jsonl"

SUBMIT_PATH = "/api/webhooks/phone-agent"
DEFAULT_RESOLVE_PATH = "/api/webhooks/phone-agent/resolve"

# The resolve call sits between the caller saying hello and us answering, so it
# gets a tight budget. The submit call runs during shutdown, where the worker
# gives us more room.
RESOLVE_TIMEOUT_S = 4.0
SUBMIT_TIMEOUT_S = 15.0
SUBMIT_ATTEMPTS = 3


@dataclass(frozen=True)
class Student:
    """A participant record the backend matched to the caller's phone number."""

    participant_id: str
    display_name: str
    consent_on_file: bool = False


class CardinalClient:
    """Talks to the phone-agent webhook. Never raises at the call sites."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        resolve_path: str | None = None,
    ) -> None:
        base = base_url if base_url is not None else os.getenv("CARDINAL_API_BASE", "")
        self._base_url = base.rstrip("/")
        self._api_key = api_key if api_key is not None else os.getenv("CARDINAL_API_KEY", "")
        self._resolve_path = resolve_path or os.getenv(
            "CARDINAL_RESOLVE_PATH", DEFAULT_RESOLVE_PATH
        )
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

    async def resolve_caller(self, phone_number: str | None) -> Student | None:
        """Match a caller ID to a student. None means 'treat this call as anonymous'.

        None covers every failure mode on purpose -- unknown number, endpoint not
        deployed yet, backend down, request timed out. The caller experience is
        identical in all of them and no record is ever attached on a guess.
        """
        if not phone_number or not self.enabled:
            return None

        url = f"{self._base_url}{self._resolve_path}"
        try:
            async with self._client().post(
                url,
                json={"phoneNumber": phone_number},
                timeout=aiohttp.ClientTimeout(total=RESOLVE_TIMEOUT_S),
            ) as resp:
                if resp.status == 404:
                    logger.info("no participant matches caller %s -- anonymous call", phone_number)
                    return None
                if resp.status != 200:
                    body = (await resp.text())[:400]
                    logger.warning(
                        "resolve failed (%s) for %s -- falling back to anonymous: %s",
                        resp.status,
                        phone_number,
                        body,
                    )
                    return None
                payload = await resp.json()
        except asyncio.TimeoutError:
            logger.warning("resolve timed out after %.1fs -- anonymous call", RESOLVE_TIMEOUT_S)
            return None
        except Exception:
            logger.exception("resolve errored -- anonymous call")
            return None

        student = _parse_student(payload)
        if student is None:
            logger.warning("resolve returned 200 with an unusable body: %r", payload)
        else:
            logger.info(
                "caller %s resolved to %s (%s)",
                phone_number,
                student.display_name,
                student.participant_id,
            )
        return student

    async def submit_transcript(
        self,
        *,
        phone_number: str,
        transcript: str,
        call_duration_sec: int,
        call_status: str | None = None,
    ) -> bool:
        """POST a finished call. Returns whether the backend accepted it.

        A rejection is not retried -- 4xx means the payload or the key is wrong and
        sending it again changes nothing. Transport failures get a few attempts,
        then the payload lands in the dead-letter file.
        """
        payload: dict[str, Any] = {
            "phoneNumber": phone_number,
            "transcript": transcript,
            "callDurationSec": call_duration_sec,
        }
        if call_status is not None:
            payload["callStatus"] = call_status

        if not self.enabled:
            # Print what would have gone over the wire. Without this a test call
            # against no backend leaves nothing to inspect, which is precisely when
            # you most want to read the transcript.
            logger.info(
                "no backend configured -- would have submitted %ss of call for %s:\n%s",
                call_duration_sec,
                phone_number,
                transcript,
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
                        logger.info(
                            "transcript accepted for %s (%ss, %d chars)",
                            phone_number,
                            call_duration_sec,
                            len(transcript),
                        )
                        return True

                    body = (await resp.text())[:400]
                    last_error = f"http {resp.status}: {body}"

                    if 400 <= resp.status < 500:
                        # Terminal. 401 is a bad key, 404 is an unmatched number,
                        # 400 is a payload we built wrong -- none improve on retry.
                        logger.error("transcript rejected, not retrying -- %s", last_error)
                        _dead_letter(payload, last_error)
                        return False

                    logger.warning("transcript submit attempt %d failed -- %s", attempt, last_error)
            except asyncio.TimeoutError:
                last_error = f"timeout after {SUBMIT_TIMEOUT_S}s"
                logger.warning("transcript submit attempt %d timed out", attempt)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("transcript submit attempt %d errored -- %s", attempt, last_error)

            if attempt < SUBMIT_ATTEMPTS:
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

        logger.error("transcript undeliverable after %d attempts -- %s", SUBMIT_ATTEMPTS, last_error)
        _dead_letter(payload, last_error)
        return False


def _parse_student(payload: Any) -> Student | None:
    """Read a resolve response. Tolerates the endpoint not existing as specified yet."""
    if not isinstance(payload, dict):
        return None

    participant_id = payload.get("participantId") or payload.get("participant_id")
    display_name = payload.get("displayName") or payload.get("display_name")
    if not participant_id or not display_name:
        return None

    return Student(
        participant_id=str(participant_id),
        display_name=str(display_name),
        # Absent means "assume we have never asked", which is the safe reading:
        # worst case a returning student hears the full consent script twice.
        consent_on_file=bool(payload.get("consentOnFile") or payload.get("consent_on_file")),
    )


def _dead_letter(payload: dict[str, Any], error: str) -> None:
    """Persist a call we could not deliver so it can be replayed by hand."""
    try:
        record = {"failed_at": time.time(), "error": error, "payload": payload}
        with DEAD_LETTER_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.error("wrote undelivered call to %s", DEAD_LETTER_PATH)
    except Exception:
        # Last resort: the transcript at least exists in the logs.
        logger.exception("could not write dead-letter record; payload was %r", payload)
