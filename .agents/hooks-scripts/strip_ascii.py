#!/usr/bin/env python3
"""
PreToolUse / PreToolExecution Hook Handler Script for Google Antigravity & Gemini CLI.
Interprets file write tool calls via stdin and silently strips all non-ASCII, non-printable
characters from newly inserted file content fields (ReplacementContent, CodeContent, content, code)
before writing, while preserving TargetContent as-is so string matching in replace operations doesn't break.
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


def main():
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            print(json.dumps({"decision": "allow"}))
            return

        payload = json.loads(raw_input)
        
        # Support both Gemini CLI format (tool_input / toolCall) and ADK format (tool_call)
        tool_call = payload.get("tool_call") or payload.get("toolCall") or {}
        args = tool_call.get("args") or payload.get("tool_input") or payload.get("args") or {}

        sanitized_args = sanitize_args(args)

        response = {
            "decision": "allow",
            "tool_input": sanitized_args,
            "hookSpecificOutput": {
                "tool_input": sanitized_args
            }
        }
        print(json.dumps(response))

    except Exception as e:
        # On error, log to stderr and fail open allowing unchanged arguments
        sys.stderr.write(f"strip_ascii hook error: {e}\n")
        print(json.dumps({"decision": "allow"}))


if __name__ == "__main__":
    main()
