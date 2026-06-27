"""Telegram notification helper for the CRN exploration system.

Sends alerts to a Telegram group/chat when important events occur:
  - DB checkpoint triggered (compound milestone)
  - New kinetically-relevant compound discovered
  - Batch crash rate spike
  - Backend/database errors

Env vars:
  TELEGRAM_BOT_TOKEN  — from @BotFather
  TELEGRAM_CHAT_ID    — group or user chat ID (negative for groups)

If either is unset, all send calls are silently skipped.
"""

import os
import time
from typing import Optional

from loguru import logger

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Rate-limit: don't send the same event type more than once per N seconds
_rate_limit: dict[str, float] = {}
DEFAULT_COOLDOWN_S = 60.0


def _is_configured() -> bool:
    return bool(BOT_TOKEN) and bool(CHAT_ID)


def send(
    text: str,
    event_key: Optional[str] = None,
    cooldown_s: float = DEFAULT_COOLDOWN_S,
) -> bool:
    """Send a Telegram message. Returns True if sent, False if skipped.

    event_key: optional dedup key for rate-limiting. If the same key was
        sent within cooldown_s, the message is suppressed.
    """
    if not _is_configured():
        return False

    if event_key:
        last = _rate_limit.get(event_key, 0.0)
        if time.time() - last < cooldown_s:
            return False

    try:
        import requests
        resp = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                "chat_id": CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        if resp.status_code == 200 and resp.json().get("ok"):
            if event_key:
                _rate_limit[event_key] = time.time()
            return True
        else:
            logger.warning(f"Telegram send failed: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        logger.warning(f"Telegram send error: {e}")
        return False


# ---- Convenience senders for specific event types ----

def notify_checkpoint(n_compounds: int, uri: str, success: bool, detail: str = ""):
    tag = "checkpoint_ok" if success else "checkpoint_fail"
    icon = "\u2705" if success else "\u274c"
    msg = (
        f"{icon} <b>DB Checkpoint</b>\n"
        f"Compounds: {n_compounds}\n"
        f"Backup: <code>{uri}</code>\n"
    )
    if detail:
        msg += f"{detail}\n"
    send(msg, event_key=tag, cooldown_s=30)


def notify_new_compound(smiles: str, formula: str, n_total: int,
                        discovery_method: str = ""):
    msg = (
        f"\U0001f9ea <b>New Compound</b>\n"
        f"<code>{smiles}</code> ({formula})\n"
        f"Total: {n_total} | via {discovery_method}"
    )
    send(msg, event_key=f"compound_{smiles}", cooldown_s=300)


def notify_error(source: str, error: str):
    msg = (
        f"\u26a0\ufe0f <b>Error: {source}</b>\n"
        f"<code>{error[:500]}</code>"
    )
    send(msg, event_key=f"error_{source}", cooldown_s=120)


def notify_workers_offline():
    msg = (
        f"\U0001f6a8 <b>All GPU workers offline</b>\n"
        f"No exploration heartbeats in the last 20 minutes."
    )
    send(msg, event_key="workers_offline", cooldown_s=600)


def notify_batch_crashes(crash_rate: float, window: int):
    msg = (
        f"\u26a0\ufe0f <b>High batch crash rate</b>\n"
        f"{crash_rate*100:.0f}% of last {window} batches failed (survived=0)"
    )
    send(msg, event_key="batch_crashes", cooldown_s=300)


def notify_hourly_status(
    n_compounds: int, n_reactions: int, n_reactions_gen: int,
    n_reactions_pes: int, n_workers: int, n_batches: int,
    merge_valid: int = 0, single_valid: int = 0,
):
    msg = (
        f"\U0001f4ca <b>Hourly Status</b>\n"
        f"Compounds: {n_compounds} | Reactions: {n_reactions}\n"
        f"  gen: {n_reactions_gen} (merge {merge_valid}, single {single_valid})\n"
        f"  PES: {n_reactions_pes}\n"
        f"Workers: {n_workers} GPU | Batches: {n_batches}"
    )
    send(msg, event_key="hourly_status", cooldown_s=3500)
