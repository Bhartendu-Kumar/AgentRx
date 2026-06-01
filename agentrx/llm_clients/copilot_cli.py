"""LLM client that delegates to the GitHub Copilot CLI.

Usage:
    Set AGENT_VERIFY_ENDPOINT_TYPE=copilot in .env (or pass --endpoint copilot).
    Requires the GitHub Copilot CLI on PATH and authenticated via `gh auth login`.

This module exposes two interfaces so it can be used as a drop-in alongside
the Azure and TRAPI clients:

    1. CopilotCLIClient  — a thin wrapper with .chat.completions.create(...)
       that mimics the AzureOpenAI client interface used everywhere in AgentRx.
    2. LLMAgent           — mirrors llm_clients.azure.LLMAgent so the Judge
       module can inherit from it.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from types import SimpleNamespace
from typing import List, Dict, Optional

import agentrx.pipeline.globals as g

try:
    import agentrx.reports.metrics as metrics
except ImportError:
    metrics = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLI_VERIFIED = False
_COPILOT_BIN: Optional[str] = None  # resolved absolute path


def _refresh_path():
    """On Windows, refresh os.environ['PATH'] from the registry so that
    binaries installed after Python started are discoverable."""
    if sys.platform == "win32":
        import winreg
        parts = []
        for hive, key in [
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            (winreg.HKEY_CURRENT_USER, r"Environment"),
        ]:
            try:
                with winreg.OpenKey(hive, key) as k:
                    val, _ = winreg.QueryValueEx(k, "Path")
                    parts.append(val)
            except OSError:
                pass
        if parts:
            os.environ["PATH"] = ";".join(parts)


def _find_copilot_bin() -> str:
    """Locate the real copilot binary, refreshing PATH if needed.

    Cross-platform binary resolution:

    On **Windows** there can be up to 4 different "copilot" entries on PATH:
      1. VS Code bootstrapper (copilot.ps1 / .bat) — interactive, avoid
      2. WinGet bootstrapper (copilot.exe ~100MB) — hangs without WinGet
         runtime context, avoid
      3. npm-installed shell script (copilot, no extension) — fine on POSIX
      4. npm-installed cmd wrapper (copilot.cmd) — fine on Windows, this
         is what we prefer

    On **macOS / Linux** the npm global install puts a plain `copilot`
    shell wrapper into one of:
      - /usr/local/bin/copilot                    (Intel Mac, system npm)
      - /opt/homebrew/bin/copilot                 (Apple Silicon Homebrew)
      - $HOME/.npm-global/bin/copilot             (user-prefix npm)
      - $HOME/.nvm/versions/node/*/bin/copilot    (nvm-managed node)

    Priority order (all platforms):
      a) AGENT_VERIFY_COPILOT_BIN env override (absolute path)
      b) Platform-specific npm global locations (direct file check)
      c) shutil.which() lookups
      d) Last-resort fallbacks (with warnings on Windows)
    """
    import shutil

    # Always refresh PATH from registry first — Node/npm/winget installs
    # done after Python started won't be visible otherwise (Windows only)
    _refresh_path()

    # 1. Explicit override (works on all platforms)
    override = os.getenv("AGENT_VERIFY_COPILOT_BIN")
    if override and os.path.exists(override):
        return override

    if sys.platform == "win32":
        # 2a. Direct check for npm global install on Windows
        appdata = os.getenv("APPDATA")
        if appdata:
            npm_cmd = os.path.join(appdata, "npm", "copilot.cmd")
            if os.path.exists(npm_cmd):
                return npm_cmd

        # 3a. PATH lookup for .cmd (npm-style)
        cmd_path = shutil.which("copilot.cmd")
        if cmd_path:
            return cmd_path

        # 4a. .exe — only if not the WinGet shim
        exe_path = shutil.which("copilot.exe")
        if exe_path and "WinGet" not in exe_path:
            return exe_path
    else:
        # 2b. Direct check for common npm global locations on macOS / Linux
        home = os.path.expanduser("~")
        candidates = [
            "/usr/local/bin/copilot",
            "/opt/homebrew/bin/copilot",
            os.path.join(home, ".npm-global", "bin", "copilot"),
            os.path.join(home, ".local", "bin", "copilot"),
        ]
        # nvm: pick the highest-version node bin if present
        nvm_root = os.path.join(home, ".nvm", "versions", "node")
        if os.path.isdir(nvm_root):
            try:
                nodes = sorted(os.listdir(nvm_root), reverse=True)
                for node_ver in nodes:
                    candidates.append(
                        os.path.join(nvm_root, node_ver, "bin", "copilot")
                    )
            except OSError:
                pass
        for c in candidates:
            if os.path.exists(c) and os.access(c, os.X_OK):
                return c

    # Fallback to whatever "copilot" resolves to via PATH (any OS)
    path = shutil.which("copilot")
    if path:
        return path

    # Absolute last resort: WinGet exe (will likely hang, but better to
    # surface a real error than silently fail)
    if sys.platform == "win32":
        exe_path = shutil.which("copilot.exe")
        if exe_path:
            print(f"[CopilotCLI] Warning: only WinGet shim found at {exe_path}; "
                  f"install via `npm i -g @github/copilot` for reliability",
                  file=sys.stderr)
            return exe_path

    raise RuntimeError(
        "'copilot' CLI not found on PATH.\n"
        "Install: npm install -g @github/copilot\n"
        "Or set AGENT_VERIFY_COPILOT_BIN to the full path."
    )


def _verify_cli():
    """Check once that the copilot binary is reachable.

    Cold-start of copilot.exe can take 30-60s on first invocation (loading
    Node, auth check, etc.), so we use a generous timeout. If --version
    hangs, we still mark the binary as verified — _call_cli has its own
    timeout and will surface real failures there.
    """
    global _CLI_VERIFIED, _COPILOT_BIN
    if _CLI_VERIFIED:
        return
    _COPILOT_BIN = _find_copilot_bin()
    try:
        result = subprocess.run(
            [_COPILOT_BIN, "--version"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,  # never let CLI block on stdin
            timeout=60,
        )
        if result.returncode == 0:
            print(f"[CopilotCLI] Found: {result.stdout.strip()}")
        else:
            print("[CopilotCLI] Warning: copilot CLI returned non-zero on --version",
                  file=sys.stderr)
    except FileNotFoundError:
        raise RuntimeError(
            f"'copilot' binary not executable at: {_COPILOT_BIN}"
        )
    except subprocess.TimeoutExpired:
        # Don't abort the whole pipeline just because --version is slow.
        # Real LLM calls have their own (longer) timeouts.
        print(f"[CopilotCLI] Warning: --version timed out; continuing anyway "
              f"(binary at {_COPILOT_BIN})", file=sys.stderr)
    _CLI_VERIFIED = True


def _flatten_messages(messages: List[Dict[str, str]]) -> str:
    """Convert OpenAI-style chat messages into a single prompt string."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "system":
            parts.append(f"<SYSTEM_INSTRUCTIONS>\n{content}\n</SYSTEM_INSTRUCTIONS>")
        elif role == "user":
            parts.append(f"<USER_MESSAGE>\n{content}\n</USER_MESSAGE>")
        elif role == "assistant":
            parts.append(f"<ASSISTANT_RESPONSE>\n{content}\n</ASSISTANT_RESPONSE>")
        elif role == "tool":
            parts.append(f"<TOOL_OUTPUT>\n{content}\n</TOOL_OUTPUT>")
    return "\n\n".join(parts)


COPILOT_TIMEOUT = int(os.getenv("AGENT_VERIFY_COPILOT_TIMEOUT", "600"))
# Model selection for Copilot CLI. Prefers AGENT_VERIFY_COPILOT_MODEL (standard naming),
# falls back to COPILOT_MODEL for backward compat. See `copilot --help` for supported
# models (e.g. claude-opus-4.6, claude-sonnet-4.5, gpt-5, gpt-5-mini).
COPILOT_MODEL = (
    os.getenv("AGENT_VERIFY_COPILOT_MODEL")
    or os.getenv("COPILOT_MODEL")
    or "claude-opus-4.6"
)


# Max characters safe for the -p flag on Windows (WinError 206 above this).
# Measured via binary search: fails at ~32,687, safe at ~32,281.
# However, stdin is actually faster (~18s vs ~20s) and has NO size limit
# (tested up to 200K+ chars), so we always use stdin.
_MAX_ARG_CHARS = 30_000  # kept for reference; stdin used for everything


def _call_cli(prompt: str, timeout: int = None) -> str:
    """Shell out to the copilot binary and return raw stdout.

    Always pipes the prompt via stdin to avoid Windows command-line length
    limits (WinError 206 at ~32K chars via -p flag).  Stdin is also ~10%
    faster and handles 200K+ chars with no issues (~18-20s constant).
    """
    timeout = timeout or COPILOT_TIMEOUT
    _verify_cli()  # ensures _COPILOT_BIN is set
    binary = _COPILOT_BIN or "copilot"

    cmd = [binary, "-s", "--no-ask-user", "--allow-all",
           "--output-format", "text"]
    if COPILOT_MODEL:
        cmd.extend(["--model", COPILOT_MODEL])

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

        if result.returncode == 0 and result.stdout and result.stdout.strip():
            return result.stdout.strip()

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()[:300]
            print(f"[CopilotCLI] exit {result.returncode}: {stderr}", file=sys.stderr)

    except subprocess.TimeoutExpired:
        print(f"[CopilotCLI] Timed out after {timeout}s", file=sys.stderr)
    except Exception as e:
        print(f"[CopilotCLI] Error: {e}", file=sys.stderr)

    return ""


# ---------------------------------------------------------------------------
# Drop-in client that quacks like AzureOpenAI
# ---------------------------------------------------------------------------

class _Completions:
    """Mimics openai.resources.chat.Completions so that
    ``client.chat.completions.create(model=..., messages=...)``
    works identically to the Azure / TRAPI path."""

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from Copilot CLI response that may contain markdown
        fences, preamble text, or other wrapper content.

        Tries (in order):
          1. ```json ... ``` fenced block
          2. ``` ... ``` generic fenced block
          3. First { or [ to last } or ]  (greedy)
          4. Original text as-is
        """
        import re

        if not text or not text.strip():
            return text

        # 1. ```json ... ```
        m = re.search(r'```json\s*\n?([\s\S]*?)```', text)
        if m:
            return m.group(1).strip()

        # 2. ``` ... ```
        m = re.search(r'```\s*\n?([\s\S]*?)```', text)
        if m:
            candidate = m.group(1).strip()
            if candidate and candidate[0] in '{[':
                return candidate

        # 3. First { or [ ... last } or ]
        first = None
        for i, c in enumerate(text):
            if c in '{[':
                first = i
                break
        if first is not None:
            last = None
            target = '}' if text[first] == '{' else ']'
            for i in range(len(text) - 1, first, -1):
                if text[i] == target:
                    last = i
                    break
            if last is not None:
                return text[first:last + 1]

        # 4. Return as-is
        return text

    def create(self, *, model: str = "", messages: list = None, **kwargs):
        _verify_cli()
        prompt = _flatten_messages(messages or [])

        # If caller requests JSON output, add a hint to the prompt
        response_format = kwargs.get("response_format", None)
        wants_json = (isinstance(response_format, dict)
                      and response_format.get("type") == "json_object")
        if wants_json:
            prompt += "\n\nIMPORTANT: Respond with ONLY valid JSON. No markdown fences, no explanation text."

        start = time.perf_counter()
        text = _call_cli(prompt)
        elapsed = time.perf_counter() - start

        # If JSON was requested, extract it from any wrapper text
        if wants_json and text:
            text = self._extract_json(text)

        # Build a response object shaped like the OpenAI SDK response
        choice = SimpleNamespace(
            message=SimpleNamespace(content=text, role="assistant"),
            index=0,
            finish_reason="stop",
        )
        usage = SimpleNamespace(
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(text) // 4,
            total_tokens=(len(prompt) + len(text)) // 4,
        )
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "copilot-cli",
            _elapsed=elapsed,
        )


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class CopilotCLIClient:
    """Mimics ``openai.AzureOpenAI`` just enough for AgentRx."""

    def __init__(self, **_kwargs):
        _verify_cli()
        self.chat = _Chat()


def copilot_mk_client(**_kwargs) -> CopilotCLIClient:
    """Factory — same call signature as ``LLMAgentAzure.azure_mk_client()``."""
    return CopilotCLIClient()


# ---------------------------------------------------------------------------
# LLMAgent subclass used by judge.py  (it inherits from the agent class)
# ---------------------------------------------------------------------------

class LLMAgent:
    """Drop-in replacement for ``llm_clients.azure.LLMAgent`` and
    ``llm_clients.trapi.LLMAgent`` so the Judge can subclass it."""

    def __init__(self, api_version=None, model_name=None,
                 model_version=None, deployment_name=None, **_kwargs):
        self.api_version = api_version or ""
        self.model_name = model_name or "copilot-cli"
        self.model_version = model_version or ""
        self.deployment_name = deployment_name or ""
        self.endpoint = "copilot-cli"
        self.last_call_telemetry = None
        self.client = CopilotCLIClient()

    def get_llm_response(self, messages):
        start_timestamp = datetime.now()
        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        end_time = time.perf_counter()
        end_timestamp = datetime.now()
        execution_time_sec = round(end_time - start_time, 4)

        observed_snapshot.record(
            served_model=getattr(response, "model", None),
            system_fingerprint=getattr(response, "system_fingerprint", None),
        )

        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        if metrics:
            token_usage = metrics.TokenUsage(
                prompt_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            time_info = metrics.TimingInfo(
                start_time=start_timestamp,
                end_time=end_timestamp,
                execution_time_sec=execution_time_sec,
            )
            self.last_call_telemetry = metrics.LLMCallTelemetry(
                tokens=token_usage,
                time=time_info,
                model_name=self.model_name,
                instance=self.endpoint,
            )

        return response

    @staticmethod
    def copilot_mk_client() -> CopilotCLIClient:
        return copilot_mk_client()
