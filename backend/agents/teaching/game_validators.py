"""HTML / JS static validators used by the Game Agent's PTC tools."""
from __future__ import annotations

import asyncio
import re

# Tags that don't need a closing tag in HTML5.
_VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}

# Tags whose content is parsed as raw text (no nested tag balancing inside).
_RAW_CONTENT_TAGS = {"script", "style"}

_TAG_RE = re.compile(
    r"<\s*(/?)\s*([a-zA-Z][a-zA-Z0-9]*)\b[^>]*?(/?)\s*>",
)


def _tag_balance_errors(html: str) -> list[str]:
    """Walk the HTML, ignoring void/raw-content tags, and report unmatched tags."""
    errors: list[str] = []
    stack: list[tuple[str, int]] = []  # (tag_name, position)
    pos = 0
    while pos < len(html):
        # Skip raw-content blocks (script/style) — anything inside is opaque.
        m = re.search(
            r"<\s*(script|style)\b[^>]*?>",
            html[pos:], re.IGNORECASE,
        )
        next_tag = _TAG_RE.search(html, pos)
        if next_tag is None:
            break
        # If a raw block opens before the next ordinary tag, skip past its closer.
        if m and pos + m.start() <= next_tag.start():
            raw_tag = m.group(1).lower()
            close_re = re.compile(rf"</\s*{raw_tag}\s*>", re.IGNORECASE)
            close_m = close_re.search(html, pos + m.end())
            if close_m is None:
                errors.append(f"Unclosed <{raw_tag}> block")
                return errors
            pos = close_m.end()
            continue

        is_close = next_tag.group(1) == "/"
        tag = next_tag.group(2).lower()
        self_close = next_tag.group(3) == "/"
        pos = next_tag.end()
        if tag in _VOID_TAGS or self_close:
            continue
        if not is_close:
            stack.append((tag, next_tag.start()))
        else:
            if not stack:
                errors.append(f"Closing </{tag}> with no matching open")
                continue
            opened, _ = stack.pop()
            if opened != tag:
                errors.append(
                    f"Mismatched close </{tag}>; expected </{opened}>"
                )
                # Try to recover by popping the closer-than-it-should-be tag
                # so subsequent pairs still report cleanly.
    if stack:
        for opened, _ in stack:
            errors.append(f"Unclosed <{opened}>")
    return errors


def _archetype_errors(html: str, archetype: str) -> list[str]:
    """Soft check that the chosen archetype's expected elements appear."""
    h = html.lower()
    issues: list[str] = []
    if archetype == "grid":
        if "<canvas" not in h and not re.search(
            r"display\s*:\s*grid", h,
        ):
            issues.append(
                "archetype=grid expects either a <canvas> or CSS "
                "`display: grid` somewhere in the markup."
            )
    elif archetype == "drag_drop":
        if "draggable" not in h:
            issues.append(
                "archetype=drag_drop expects at least one element with "
                "the `draggable` attribute."
            )
    elif archetype == "button_array":
        button_count = len(re.findall(r"<button\b", h))
        if button_count < 2:
            issues.append(
                f"archetype=button_array expects at least 2 <button> "
                f"elements; found {button_count}."
            )
    elif archetype == "canvas":
        if "<canvas" not in h:
            issues.append("archetype=canvas expects a <canvas> element.")
    elif archetype == "list":
        if "<ol" not in h and "<ul" not in h:
            issues.append("archetype=list expects an <ol> or <ul> element.")
    return issues


def validate_html_string(html: str, *, archetype: str | None = None) -> list[str]:
    """Returns list of error strings; empty list means valid."""
    errors: list[str] = []
    errors.extend(_tag_balance_errors(html))
    if archetype is not None:
        errors.extend(_archetype_errors(html, archetype))
    return errors


async def validate_js_string(js: str, *, timeout: float = 8.0) -> str | None:
    """`node --check` on a JS body. Returns error text or None on success.

    No-op (returns None) if `node` isn't on PATH so tests don't require it.
    """
    if not js.strip():
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "node", "--check", "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=js.encode("utf-8")), timeout=timeout,
        )
    except (TimeoutError, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return (stderr or stdout).decode("utf-8", errors="replace").strip()
    return None
