"""
Agente principal.

Implementa o loop de raciocínio:

    pergunta_usuario
        |
        v
    [LLM raciocina sobre o que fazer]
        |
        v
    LLM gerou tool_call? --- sim ---> [executa tool] --- adiciona resultado ---+
        |                                                                      |
        | não (stop_reason == 'end_turn')                                       |
        v                                                                      |
    [resposta final em texto]                                                  |
        ^                                                                      |
        |                                                                      |
        +----------------------------------------------------------------------+

Limitamos a MAX_AGENT_ITERATIONS para evitar loops infinitos.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import json

from .llm_client import LLMClient, LLMResponse
from tools import all_tools_for_llm, get_tool_by_name, state
from config import MAX_AGENT_ITERATIONS


# ============================================================
# Estruturas para registrar a trajetória do agente
# ============================================================

@dataclass
class Step:
    """Um passo da trajetória: ou um pensamento do LLM, ou uma chamada de tool."""
    tipo: str                              # 'llm_text' | 'tool_call' | 'tool_result'
    conteudo: dict | str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentResult:
    """Resultado completo de uma pergunta processada pelo agente."""
    pergunta: str
    resposta_final: str
    sucesso: bool                          # Terminou normalmente?
    trajetoria: list[Step]                 # Cada passo intermediário
    total_iteracoes: int
    total_tool_calls: int
    input_tokens: int
    output_tokens: int
    latencia_total: float

    def to_dict(self) -> dict:
        return {
            "pergunta": self.pergunta,
            "resposta_final": self.resposta_final,
            "sucesso": self.sucesso,
            "total_iteracoes": self.total_iteracoes,
            "total_tool_calls": self.total_tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latencia_total": round(self.latencia_total, 3),
            "trajetoria": [
                {"tipo": s.tipo, "conteudo": s.conteudo, "timestamp": s.timestamp}
                for s in self.trajetoria
            ],
        }


# ============================================================
# Prompt de sistema
# ============================================================

SYSTEM_PROMPT = """\
Você é um assistente de análise exploratória de dados.

Sua tarefa é responder perguntas em português sobre um arquivo CSV que está
carregado em memória. Você NÃO tem acesso direto aos dados — você precisa
chamar as ferramentas (tools) disponíveis para inspecionar e operar sobre
o dataset.

Diretrizes:
1. Sempre que receber uma pergunta nova, considere chamar listar_colunas()
   primeiro se ainda não conhecer a estrutura do dataset.
2. Use os NOMES EXATOS das colunas como retornados por listar_colunas().
   Não invente colunas — se uma coluna mencionada pelo usuário não existir,
   peça esclarecimento ou avise.
3. Se a pergunta for ambígua ou impossível de responder com os dados
   disponíveis, diga isso explicitamente em vez de inventar uma resposta.
4. Use o menor número de tool calls necessário. Não chame tools redundantes.
5. Quando responder ao usuário, seja conciso e em português claro. Apresente
   números com formato legível (ex.: 1.234,56 em vez de 1234.5612).
6. Em perguntas factuais simples, responda somente ao que foi perguntado.
   Não adicione colunas, exemplos ou contexto extra se o usuário não pedir.
7. Em perguntas ambíguas, responda apenas " Pergunta ambígua!"
"""


# ============================================================
# Classe Agent
# ============================================================

class Agent:
    """Orquestrador do loop de raciocínio do agente."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()
        self.tools_para_llm = all_tools_for_llm()

    def _executar_tool(self, tool_call: dict) -> dict:
        """
        Executa uma tool e retorna seu resultado.

        Captura exceções para o agente nunca travar - se a tool falhar,
        o erro é devolvido ao LLM, que pode tentar outra abordagem.
        """
        nome = tool_call["name"]
        argumentos = tool_call["input"]
        spec = get_tool_by_name(nome)

        if spec is None:
            return {"erro": f"Tool '{nome}' não encontrada no registro."}

        try:
            return spec.function(**argumentos)
        except TypeError as e:
            # Argumentos errados
            return {"erro": f"Argumentos inválidos para '{nome}': {e}"}
        except Exception as e:
            # Qualquer outro erro
            return {"erro": f"Erro ao executar '{nome}': {type(e).__name__}: {e}"}

    def perguntar(self, pergunta: str) -> AgentResult:
        """
        Processa uma pergunta e retorna o resultado completo, incluindo
        toda a trajetória para análise posterior.
        """
        # Verifica se tem dataset carregado antes de começar
        try:
            state.require_loaded()
        except RuntimeError as e:
            return AgentResult(
                pergunta=pergunta,
                resposta_final=str(e),
                sucesso=False,
                trajetoria=[],
                total_iteracoes=0,
                total_tool_calls=0,
                input_tokens=0,
                output_tokens=0,
                latencia_total=0.0,
            )

        # Histórico no formato Anthropic
        messages: list[dict] = [
            {"role": "user", "content": pergunta},
        ]

        trajetoria: list[Step] = []
        total_input = 0
        total_output = 0
        latencia_total = 0.0
        total_tool_calls = 0

        # ============ Loop principal ============
        for iteracao in range(MAX_AGENT_ITERATIONS):

            resposta: LLMResponse = self.llm.chat(
                messages=messages,
                tools=self.tools_para_llm,
                system=SYSTEM_PROMPT,
            )

            total_input += resposta.input_tokens
            total_output += resposta.output_tokens
            latencia_total += resposta.latency_seconds

            # Registra o que o LLM "pensou"/disse
            if resposta.text:
                trajetoria.append(Step(tipo="llm_text", conteudo=resposta.text))

            # Caso 1: LLM terminou (sem tool_use) — temos a resposta final
            if resposta.stop_reason == "end_turn" or not resposta.tool_calls:
                return AgentResult(
                    pergunta=pergunta,
                    resposta_final=resposta.text or "(sem resposta)",
                    sucesso=True,
                    trajetoria=trajetoria,
                    total_iteracoes=iteracao + 1,
                    total_tool_calls=total_tool_calls,
                    input_tokens=total_input,
                    output_tokens=total_output,
                    latencia_total=latencia_total,
                )

            # Caso 2: LLM quer chamar uma ou mais tools
            # Devemos adicionar ao histórico TODO o conteúdo retornado pelo LLM
            # (texto + tool_uses), e em seguida os tool_results.
            messages.append({
                "role": "assistant",
                "content": resposta.raw_response.content,
            })

            tool_results_para_llm = []
            for tc in resposta.tool_calls:
                total_tool_calls += 1
                trajetoria.append(Step(
                    tipo="tool_call",
                    conteudo={"nome": tc["name"], "argumentos": tc["input"]},
                ))

                resultado = self._executar_tool(tc)

                trajetoria.append(Step(tipo="tool_result", conteudo=resultado))

                tool_results_para_llm.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": json.dumps(resultado, ensure_ascii=False, default=str),
                })

            # Adiciona os resultados como mensagem do usuário (convenção Anthropic)
            messages.append({
                "role": "user",
                "content": tool_results_para_llm,
            })

        # Saiu do loop sem terminar — atingiu MAX_AGENT_ITERATIONS
        return AgentResult(
            pergunta=pergunta,
            resposta_final=(
                f"Limite de {MAX_AGENT_ITERATIONS} iterações atingido sem "
                "chegar a uma resposta final."
            ),
            sucesso=False,
            trajetoria=trajetoria,
            total_iteracoes=MAX_AGENT_ITERATIONS,
            total_tool_calls=total_tool_calls,
            input_tokens=total_input,
            output_tokens=total_output,
            latencia_total=latencia_total,
        )
