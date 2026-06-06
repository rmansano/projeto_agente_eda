"""
Infraestrutura comum das ferramentas (tools).

Define:
  - Estado compartilhado (o DataFrame em memória)
  - Decorador @tool que registra funções como ferramentas do agente
  - Estruturas para descrever cada tool ao LLM
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
import inspect
import pandas as pd


# ============================================================
# Estado compartilhado
# ============================================================

class DataState:
    """
    Mantém o DataFrame atual em memória.
    Todas as tools acessam o mesmo objeto, evitando recarregar o CSV
    a cada chamada.
    """
    def __init__(self) -> None:
        self.df: pd.DataFrame | None = None
        self.path: str | None = None

    def load(self, path: str) -> None:
        """Carrega um CSV no estado."""
        self.df = pd.read_csv(path)
        self.path = path

    def require_loaded(self) -> pd.DataFrame:
        """Retorna o DataFrame ou levanta erro se nada foi carregado."""
        if self.df is None:
            raise RuntimeError(
                "Nenhum dataset carregado. Carregue um CSV antes de chamar as tools."
            )
        return self.df


# Instância global - usada por todas as tools
state = DataState()


# ============================================================
# Registro de tools
# ============================================================

@dataclass
class ToolSpec:
    """Descrição de uma tool para ser passada ao LLM."""
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema dos parâmetros
    function: Callable           # A função Python que executa de fato


# Lista global de todas as tools registradas
TOOL_REGISTRY: list[ToolSpec] = []


def tool(description: str, parameters: dict[str, Any]):
    """
    Decorador que registra uma função como tool disponível para o agente.

    Uso:
        @tool(
            description="Retorna a lista de colunas e seus tipos.",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
        def listar_colunas() -> dict:
            ...

    O `description` é lido pelo LLM para decidir QUANDO chamar a tool.
    O `parameters` segue o padrão JSON Schema usado por function calling.
    """
    def decorator(func: Callable):
        spec = ToolSpec(
            name=func.__name__,
            description=description,
            parameters=parameters,
            function=func,
        )
        TOOL_REGISTRY.append(spec)
        return func
    return decorator


def get_tool_by_name(name: str) -> ToolSpec | None:
    """Busca uma tool pelo nome (usado pelo orquestrador)."""
    for spec in TOOL_REGISTRY:
        if spec.name == name:
            return spec
    return None


def all_tools_for_llm() -> list[dict]:
    """
    Retorna a lista de tools no formato que a API da Anthropic espera.

    Outros provedores (OpenAI, Google) têm formatos parecidos mas com
    diferenças sutis — adapte no llm_client.py se trocar de provedor.
    """
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in TOOL_REGISTRY
    ]
