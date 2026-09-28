import json

from openai import OpenAI

from tools.format import extract_code


class LLMClient:
    """Small OpenAI-compatible client used only by optional Executor agents."""

    def __init__(self, base_url, api_key):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def call(
        self,
        model_name,
        prompt,
        history=None,
        n=1,
        max_tokens=2048,
        temperature=0.8,
        top_p=1.0,
        top_k=40,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        timeout=180,
        system_message=None,
        need_json=False,
        **_,
    ):
        messages = list(history or [])
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )
        response = completion.choices[0].message.content.strip()
        if need_json:
            parsed = json.loads(extract_code(response))
            if not isinstance(parsed, (dict, list)):
                raise ValueError("LLM response is not a JSON object or array")
        messages.append({"role": "assistant", "content": response})
        return response, messages
