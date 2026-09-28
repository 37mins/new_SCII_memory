import json
import os
import re
import time
from typing import Any, Dict, List, Optional


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _shorten(value: Any, limit: int) -> str:
    text = _to_text(value)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


class SC2ReflectionDistiller:
    """Optional LLM reflection distiller for SC2 memory assets.

    The distiller intentionally sits behind an explicit switch. If the OpenAI
    client package, model, or API key is unavailable, callers can fall back to
    stricter rule-based assets without breaking training.
    """

    def __init__(
        self,
        enabled: bool = False,
        model: str = "",
        api_key: str = "",
        api_base: str = "",
        timeout: float = 30.0,
        temperature: float = 0.2,
        max_tokens: int = 900,
        log_path: str = "",
    ):
        self.enabled = _as_bool(enabled)
        self.model = model or os.getenv("SC2_REFLECTION_MODEL", "")
        self.api_key = api_key or os.getenv("SC2_REFLECTION_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        self.api_base = api_base or os.getenv("SC2_REFLECTION_API_BASE", "") or os.getenv("OPENAI_BASE_URL", "")
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.log_path = log_path or ""
        self._client = None
        self._client_error = ""
        if self.enabled and self.model and self.api_key:
            self._init_client()

    def _init_client(self) -> None:
        try:
            from openai import OpenAI  # type: ignore

            kwargs = {"api_key": self.api_key, "timeout": self.timeout}
            if self.api_base:
                kwargs["base_url"] = self.api_base
            self._client = OpenAI(**kwargs)
        except Exception as exc:
            self._client = None
            self._client_error = str(exc)

    def is_available(self) -> bool:
        return bool(self.enabled and self.model and self.api_key and self._client is not None)

    def distill(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "used_llm": False}
        if not self.is_available():
            return {
                "status": "unavailable",
                "used_llm": False,
                "error": self._client_error or "missing reflection model or api key",
            }

        prompt = self._build_prompt(scene)
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scene_id": scene.get("scene_id", ""),
            "status": "request",
            "prompt": prompt,
        }
        _append_jsonl(self.log_path, row)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You compress StarCraft II training episodes into short, reusable decision assets. "
                            "Output strict JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = response.choices[0].message.content or ""
            parsed = self._extract_json(raw)
            if parsed is None:
                result = {"status": "parse_error", "used_llm": True, "raw_response": _shorten(raw, 4000)}
            else:
                result = {"status": "success", "used_llm": True, "reflection": parsed, "raw_response": _shorten(raw, 4000)}
        except Exception as exc:
            result = {"status": "error", "used_llm": True, "error": str(exc)}

        _append_jsonl(
            self.log_path,
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scene_id": scene.get("scene_id", ""),
                "status": result.get("status", ""),
                "result": result,
            },
        )
        return result

    def _build_prompt(self, scene: Dict[str, Any]) -> str:
        scene_json = json.dumps(scene, ensure_ascii=False, default=str)
        return (
            "Compress this SC2 transition into reusable memory assets for future build-order decisions.\n"
            "Only keep information that can change future actions. Avoid raw history, exact timestamps, and verbose narration.\n\n"
            "Rules:\n"
            "- Emit at most 2 world_facts and 2 action_skills.\n"
            "- world_facts are environment rules or constraints, not one-off observations.\n"
            "- action_skills must be reusable 2-5 step chains or clear negative boundaries.\n"
            "- Do not create a do skill from a single ordinary action such as scv, marine, supplydepot, or barracks.\n"
            "- Do not create an avoid skill from only 'episode ended without win'; require a specific failure boundary.\n"
            "- Generalize conditions, for example army < 8, supply_left = 0, minerals < 150, or army >= 8.\n"
            "- Use polarity values exactly: do, avoid, diagnostic.\n"
            "- Use confidence in [0, 1]. Keep confidence below 0.65 if the asset is not replayable.\n\n"
            "Return JSON with this schema:\n"
            "{\n"
            '  "summary": "short scene summary",\n'
            '  "selected_evidence": ["short evidence strings"],\n'
            '  "world_facts": [\n'
            '    {"fact": "...", "condition": "...", "tags": ["..."], "confidence": 0.0, "quality_rationale": "..."}\n'
            "  ],\n"
            '  "action_skills": [\n'
            '    {"polarity": "do|avoid|diagnostic", "intent": "...", "condition": "...", '
            '"action_chain": ["..."], "preconditions": ["..."], "failure_signals": ["..."], '
            '"tags": ["..."], "confidence": 0.0, "quality_rationale": "..."}\n'
            "  ],\n"
            '  "open_questions": ["..."]\n'
            "}\n\n"
            f"SC2 scene:\n{scene_json}"
        )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        cleaned = _to_text(text)
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        if fenced:
            cleaned = fenced.group(1)
        start = cleaned.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(cleaned)):
            char = cleaned[idx]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(cleaned[start : idx + 1])
                    except Exception:
                        return None
                    return value if isinstance(value, dict) else None
        return None
