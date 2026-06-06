"""
Cliente da API do LLM.

Versão adaptada para DeepSeek usando API compatível com OpenAI.
Mantém a mesma interface esperada por agent/agent.py:
    - classe LLMClient
    - dataclass LLMResponse
    - método chat(messages, tools, system)

"""

from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
import json
import time

from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    LLM_MODEL,
    MAX_TOKENS_PER_RESPONSE,
)


# ============================================================
# Estruturas de retorno
# ============================================================

@dataclass
class LLMResponse:
    """Resposta tipada do LLM, independente do provedor."""
    text: str
    tool_calls: list[dict]
    raw_response: object
    input_tokens: int
    output_tokens: int
    stop_reason: str
    latency_seconds: float


# ============================================================
# Cliente DeepSeek
# ============================================================

class LLMClient:
    """Cliente que se comunica com a API da DeepSeek."""

    def __init__(self, model: str = LLM_MODEL, api_key: str = DEEPSEEK_API_KEY):
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY não definida. Configure no arquivo .env."
            )
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = model

    def _tools_para_openai(self, tools: list[dict]) -> list[dict]:
        """Converte tools do formato Anthropic-like do projeto para OpenAI/DeepSeek."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    def _messages_para_openai(self, messages: list[dict], system: str = "") -> list[dict]:
        """
        Converte o histórico usado pelo agent.py para o formato OpenAI/DeepSeek.

        O agent.py foi escrito no estilo Anthropic:
        - usuário inicial: string
        - resposta do assistente com tool_use: lista de blocos
        - resultado de tool: lista de blocos tool_result dentro de uma mensagem user

        A API OpenAI/DeepSeek espera:
        - mensagens normais user/assistant/system
        - assistant com tool_calls
        - uma mensagem role='tool' para cada resultado de tool
        """
        saida: list[dict] = []

        if system:
            saida.append({"role": "system", "content": system})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Caso simples: conteúdo textual.
            if isinstance(content, str):
                saida.append({"role": role, "content": content})
                continue

            # Conteúdo em blocos.
            if isinstance(content, list):
                # Mensagem do assistente contendo texto + tool_use.
                if role == "assistant":
                    texto = ""
                    tool_calls = []

                    for bloco in content:
                        if not isinstance(bloco, dict):
                            continue

                        if bloco.get("type") == "text":
                            texto += bloco.get("text", "")

                        elif bloco.get("type") == "tool_use":
                            args = bloco.get("input", {}) or {}
                            tool_calls.append({
                                "id": bloco.get("id") or bloco.get("name"),
                                "type": "function",
                                "function": {
                                    "name": bloco.get("name"),
                                    "arguments": json.dumps(args, ensure_ascii=False),
                                },
                            })

                    mensagem_assistente = {
                        "role": "assistant",
                        "content": texto or None,
                    }
                    if tool_calls:
                        mensagem_assistente["tool_calls"] = tool_calls
                    saida.append(mensagem_assistente)
                    continue

                # Mensagem com resultados das tools. No agent.py ela vem como role=user,
                # mas para DeepSeek/OpenAI precisa virar role=tool.
                for bloco in content:
                    if not isinstance(bloco, dict):
                        continue

                    if bloco.get("type") == "tool_result":
                        saida.append({
                            "role": "tool",
                            "tool_call_id": bloco.get("tool_use_id"),
                            "content": bloco.get("content", ""),
                        })
                    elif bloco.get("type") == "text":
                        saida.append({"role": role, "content": bloco.get("text", "")})

        return saida

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
    ) -> LLMResponse:
        """
        Envia uma rodada de mensagens ao LLM.
        Retorna LLMResponse no formato esperado pelo Agent.
        """
        kwargs = {
            "model": self.model,
            "max_tokens": MAX_TOKENS_PER_RESPONSE,
            "messages": self._messages_para_openai(messages, system=system),
            "tools": self._tools_para_openai(tools) if tools else None,
            "tool_choice": "auto" if tools else None,
        }

        # Remove campos None para evitar incompatibilidades.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        inicio = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        latencia = time.perf_counter() - inicio

        choice = resp.choices[0]
        msg = choice.message

        texto = msg.content or ""
        tool_calls: list[dict] = []
        conteudo_para_historico: list[dict] = []

        if texto:
            conteudo_para_historico.append({"type": "text", "text": texto})

        for tc in msg.tool_calls or []:
            try:
                argumentos = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                argumentos = {}

            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "input": argumentos,
            })

            # O agent.py reaproveita raw_response.content no histórico.
            # Mantemos blocos parecidos com Anthropic para não mexer no agent.py.
            conteudo_para_historico.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": argumentos,
            })

        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        return LLMResponse(
            text=texto,
            tool_calls=tool_calls,
            raw_response=SimpleNamespace(content=conteudo_para_historico),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason="tool_use" if tool_calls else "end_turn",
            latency_seconds=latencia,
        )
