#!/usr/bin/env python3
"""
PreToolUse Hook Handler Script for Google Antigravity.
Intercepts run_command tool execution via stdin and returns JSON decision to stdout.
"""

import json
import re
import sys

DENIED_PATTERNS = [
    # Destructive System & Filesystem Commands
    (r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+(/|/\*|~|\.|\*)", "Destructive recursive force deletion (rm -rf / or root/home/wildcard)"),
    (r"\brm\s+-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+(/|/\*|~|\.|\*)", "Destructive recursive force deletion (rm -fr / or root/home/wildcard)"),
    (r"\bmkfs(\.[a-z0-9]+)?\b", "Disk formatting operation (mkfs)"),
    (r"\bdd\s+.*of=/dev/(sd|hd|nvme|mapper|vd)", "Raw block device write operation (dd)"),
    (r"\bchmod\s+-[a-zA-Z]*R\s+(000|777)\s+/", "Broad system permission override (chmod -R)"),
    (r"\bchown\s+-[a-zA-Z]*R\s+.*\s+/", "Broad system ownership override (chown -R)"),

    # Git History Destruction & Rewriting Commands
    (r"\bgit\s+push\s+.*(--force|-f|--force-with-lease|\+)", "Git force push operation"),
    (r"\bgit\s+reset\s+--hard\b", "Git hard reset (git reset --hard)"),
    (r"\bgit\s+(filter-branch|filter-repo)\b", "Git history rewriting operation"),
    (r"\bgit\s+reflog\s+expire\b", "Git reflog expiration"),
    (r"\bgit\s+gc\s+.*--prune=", "Git garbage collection prune"),
    (r"\bgit\s+branch\s+-D\b", "Force branch deletion (git branch -D)"),
    (r"\bgit\s+stash\s+(drop|clear)\b", "Git stash destruction (stash drop/clear)"),
]

def main():
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            print(json.dumps({"decision": "allow"}))
            return

        payload = json.loads(raw_input)
        tool_call = payload.get("toolCall", {})
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args", {})

        command = args.get("CommandLine") or args.get("command") or args.get("cmd") or ""

        if command:
            for pattern, description in DENIED_PATTERNS:
                if re.search(pattern, str(command), re.IGNORECASE):
                    result = {
                        "decision": "deny",
                        "reason": f"PreToolUse Hard Safety Block: {description} in command '{command}'."
                    }
                    print(json.dumps(result))
                    return

        print(json.dumps({"decision": "allow"}))
    except Exception as e:
        # On script processing error, fail safe with deny or allow
        print(json.dumps({"decision": "allow"}))

if __name__ == "__main__":
    main()
