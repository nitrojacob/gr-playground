#!/usr/bin/env python3
"""
PreToolUse / PreToolExecution Hook Handler Script for Google Antigravity & Gemini CLI.
Interprets file write tool calls via stdin.

Enforces strict Fail-Closed behavior:
If ANY non-ASCII or non-printable character is detected in inserted content fields
(ReplacementContent, CodeContent, content, code, etc.), the hook DENIES (blocks) the tool execution.
This guarantees that file writes containing icons/emojis cannot succeed, regardless of whether
the runner supports argument mutation.
"""

import json
import sys


def find_non_ascii(val, key_path="") -> list:
    """
    Recursively scans JSON value to find non-ASCII or non-printable characters.
    Exempts 'TargetContent' because TargetContent matches existing lines on disk.
    """
    violations = []
    if isinstance(val, str):
        for idx, char in enumerate(val):
            code = ord(char)
            # Allow printable ASCII (32-126) and standard whitespace (\n, \r, \t)
            if not ((32 <= code <= 126) or char in ("\n", "\r", "\t")):
                violations.append((key_path, char, f"U+{code:04X}"))
    elif isinstance(val, dict):
        for k, v in val.items():
            if k == "TargetContent":
                continue
            sub_path = f"{key_path}.{k}" if key_path else k
            violations.extend(find_non_ascii(v, sub_path))
    elif isinstance(val, list):
        for idx, item in enumerate(val):
            sub_path = f"{key_path}[{idx}]"
            violations.extend(find_non_ascii(item, sub_path))
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

        # Scan for non-ASCII / icon violations
        violations = find_non_ascii(args)
        if violations:
            sample_summary = ", ".join(f"'{c}' ({code}) at {path}" for path, c, code in violations[:5])
            msg = f"PreToolUse Security Block: Non-ASCII/icon characters detected in file write payload ({sample_summary}). All file write content must be strictly ASCII."
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


