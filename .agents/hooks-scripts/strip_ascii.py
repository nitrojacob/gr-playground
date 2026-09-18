#!/usr/bin/env python3
"""
PreToolUse / PreToolExecution Hook Handler Script for Google Antigravity & Gemini CLI.
Interprets file write tool calls via stdin and silently strips all non-ASCII, non-printable
characters from newly inserted file content fields (ReplacementContent, CodeContent, content, code)
before writing, while preserving TargetContent as-is so string matching in replace operations doesn't break.

Implements Fail-Closed behavior (returns decision: deny on parsing error or invalid arguments)
and robust argument extraction scoped specifically to file write tool calls.
"""

import json
import sys


def strip_non_ascii_printable(text: str) -> str:
    """
    Strips non-ASCII and non-printable characters.
    Preserves printable ASCII (32-126) and standard keyboard whitespace (newline \n, CR \r, tab \t).
    """
    if not isinstance(text, str):
        return text
    return "".join(c for c in text if (32 <= ord(c) <= 126) or c in ("\n", "\r", "\t"))


def sanitize_args(args: dict) -> dict:
    """
    Recursively scans and sanitizes text payload fields in tool argument dictionary.
    Exempts 'TargetContent' so existing file line matching in replace operations is preserved.
    """
    if not isinstance(args, dict):
        raise TypeError(f"Expected argument dictionary, got {type(args).__name__}")

    sanitized = {}
    for key, value in args.items():
        # DO NOT strip TargetContent - it must match existing lines on disk exactly
        if key == "TargetContent":
            sanitized[key] = value
        elif isinstance(value, str):
            sanitized[key] = strip_non_ascii_printable(value)
        elif isinstance(value, list):
            sanitized_list = []
            for item in value:
                if isinstance(item, str):
                    sanitized_list.append(strip_non_ascii_printable(item))
                elif isinstance(item, dict):
                    sanitized_list.append(sanitize_args(item))
                else:
                    sanitized_list.append(item)
            sanitized[key] = sanitized_list
        elif isinstance(value, dict):
            sanitized[key] = sanitize_args(value)
        else:
            sanitized[key] = value
    return sanitized


def extract_file_write_args(payload: dict) -> dict:
    """
    Robustly extracts argument dictionary from various payload formats specifically for file write tool calls.
    Raises ValueError if valid file write argument dictionary cannot be found.
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON dictionary.")

    # Candidate containers for tool call objects
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

    # Fallback: payload itself is a flat dictionary containing known file write keys
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
        sanitized_args = sanitize_args(args)

        response = {
            "decision": "allow",
            "tool_input": sanitized_args,
            "tool_call": {
                "args": sanitized_args
            },
            "hookSpecificOutput": {
                "tool_input": sanitized_args
            }
        }
        print(json.dumps(response))

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

