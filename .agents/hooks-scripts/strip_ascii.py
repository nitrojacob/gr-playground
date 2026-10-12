#!/usr/bin/env python3
"""
PreToolUse / PreToolExecution Hook Handler Script for Google Antigravity & Gemini CLI.
Interprets file write tool calls via stdin.

Enforces strict Fail-Closed behavior:
If ANY emoji or non-ASCII character is detected in inserted content fields
(CodeContent, ReplacementContent, content, code, etc.), the hook DENIES (blocks) the tool execution
with an informative message explicitly listing the detected emoji(s) and the exact line numbers
needing correction.

Exempts TargetContent so that string-matching in replace operations to remove existing emojis is preserved.
"""

import json
import sys
import unicodedata


def find_line_violations(text: str, start_line: int = 1) -> dict:
    """
    Scans multi-line text and maps 1-indexed line numbers to lists of non-ASCII/emoji descriptors.
    """
    violations = {}
    if not isinstance(text, str):
        return violations

    lines = text.split("\n")
    for offset, line in enumerate(lines):
        line_num = start_line + offset
        line_emojis = []
        for char in line:
            code = ord(char)
            # Check for non-ASCII or non-printable character
            if not ((32 <= code <= 126) or char in ("\t", "\r")):
                # Variation selectors (e.g. \ufe0f) can be skipped as they modify the preceding emoji
                if code == 0xFE0F:
                    continue
                name = unicodedata.name(char, "UNKNOWN")
                line_emojis.append(f"'{char}' (U+{code:04X}: {name})")
        if line_emojis:
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for item in line_emojis:
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
            violations[line_num] = deduped

    return violations


def scan_args_for_emoji_violations(args: dict) -> dict:
    """
    Scans file write argument dictionary for emoji / non-ASCII characters.
    Returns a dict mapping line_number (int) -> list of formatted emoji descriptors.
    """
    violations = {}

    # Case 1: write_to_file (CodeContent / content / code)
    for content_key in ("CodeContent", "content", "code"):
        if content_key in args and isinstance(args[content_key], str):
            res = find_line_violations(args[content_key], start_line=1)
            for l_num, items in res.items():
                violations.setdefault(l_num, []).extend(items)

    # Case 2: replace_file_content (ReplacementContent)
    if "ReplacementContent" in args and isinstance(args["ReplacementContent"], str):
        start_line = 1
        if "StartLine" in args and args["StartLine"] is not None:
            try:
                start_line = int(args["StartLine"])
            except (ValueError, TypeError):
                start_line = 1
        res = find_line_violations(args["ReplacementContent"], start_line=start_line)
        for l_num, items in res.items():
            violations.setdefault(l_num, []).extend(items)

    # Case 3: multi_replace_file_content (ReplacementChunks)
    if "ReplacementChunks" in args and isinstance(args["ReplacementChunks"], list):
        for chunk in args["ReplacementChunks"]:
            if isinstance(chunk, dict) and "ReplacementContent" in chunk and isinstance(chunk["ReplacementContent"], str):
                start_line = 1
                if "StartLine" in chunk and chunk["StartLine"] is not None:
                    try:
                        start_line = int(chunk["StartLine"])
                    except (ValueError, TypeError):
                        start_line = 1
                res = find_line_violations(chunk["ReplacementContent"], start_line=start_line)
                for l_num, items in res.items():
                    violations.setdefault(l_num, []).extend(items)

    # Case 4: Any other string field (excluding metadata and TargetContent)
    exempt_keys = {
        "TargetContent",
        "TargetFile",
        "path",
        "Description",
        "Instruction",
        "toolAction",
        "toolSummary",
        "CodeContent",
        "ReplacementContent",
        "ReplacementChunks",
        "content",
        "code",
        "StartLine",
        "EndLine",
        "Overwrite",
        "Append",
        "AllowMultiple",
        "TargetLintErrorIds",
        "ArtifactMetadata",
    }
    for key, val in args.items():
        if key not in exempt_keys:
            if isinstance(val, str):
                res = find_line_violations(val, start_line=1)
                for l_num, items in res.items():
                    violations.setdefault(l_num, []).extend(items)
            elif isinstance(val, dict):
                sub_res = scan_args_for_emoji_violations(val)
                for l_num, items in sub_res.items():
                    violations.setdefault(l_num, []).extend(items)

    return violations


def extract_file_write_args(payload: dict) -> dict:
    """
    Robustly extracts argument dictionary from various payload formats specifically for file write tool calls.
    Raises ValueError if valid file write argument dictionary cannot be found.
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON dictionary.")

    containers = [
        payload.get("tool_call"),
        payload.get("toolCall"),
        payload.get("tool"),
        payload,
    ]

    for container in containers:
        if isinstance(container, dict):
            for key in ("args", "arguments", "tool_input", "input", "parameters"):
                val = container.get(key)
                if isinstance(val, dict):
                    return val

    file_write_keys = {"TargetFile", "CodeContent", "ReplacementContent", "ReplacementChunks", "path", "content", "code"}
    if any(k in payload for k in file_write_keys):
        return payload

    raise ValueError("Could not extract argument dictionary from file write tool payload.")


def main():
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            raise ValueError("Empty input received by strip_ascii hook handler.")

        payload = json.loads(raw_input)
        args = extract_file_write_args(payload)

        # Scan for emoji / non-ASCII violations mapped by line numbers
        violations = scan_args_for_emoji_violations(args)
        if violations:
            target_file = args.get("TargetFile") or args.get("path") or "file"
            lines_summary = []
            for line_num in sorted(violations.keys()):
                # Deduplicate emojis on this line
                seen = set()
                unique_emojis = []
                for e in violations[line_num]:
                    if e not in seen:
                        seen.add(e)
                        unique_emojis.append(e)
                lines_summary.append(f"  - Line {line_num}: {', '.join(unique_emojis)}")

            details = "\n".join(lines_summary)
            msg = (
                f"PreToolUse Security Block: File write to '{target_file}' rejected because it contains emoji characters that need correction:\n"
                f"{details}\n"
                f"Please remove all emojis before writing to the file."
            )
            sys.stderr.write(f"strip_ascii block: {msg}\n")
            print(json.dumps({
                "decision": "deny",
                "reason": msg
            }))
            return

        # If clean, allow execution
        print(json.dumps({"decision": "allow"}))

    except Exception as e:
        # Fail-closed (deny) to prevent un-sanitized writes on error
        error_msg = f"strip_ascii hook processing error: {e}"
        sys.stderr.write(f"{error_msg}\n")
        print(json.dumps({
            "decision": "deny",
            "reason": f"PreToolUse Security Block (Fail-Closed): {error_msg}"
        }))


if __name__ == "__main__":
    main()



