"""
Tools extras para pontuar no critério de originalidade.

As três ferramentas abaixo complementam a EDA obrigatória:
  - contar_nulos: diagnóstico de qualidade dos dados;
  - calcular_percentil: análise de distribuição além de média/mediana;
  - tabela_contingencia: cruzamento entre duas variáveis categóricas.
"""

from __future__ import annotations
import pandas as pd
from .base import tool, state


@tool(
    description=(
        "Conta valores nulos por coluna e retorna também a porcentagem de nulos. "
        "Use para avaliar qualidade/completude do dataset."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
def contar_nulos() -> dict:
    """Retorna a quantidade e porcentagem de valores nulos por coluna."""
    df = state.require_loaded()
    total = len(df)
    por_coluna = {}
    for col in df.columns:
        qtd = int(df[col].isna().sum())
        por_coluna[col] = {
            "nulos": qtd,
            "porcentagem": round(qtd / total * 100, 2) if total else 0.0,
        }
    return {
        "total_linhas": total,
        "colunas_com_nulos": int(sum(1 for v in por_coluna.values() if v["nulos"] > 0)),
        "por_coluna": por_coluna,
    }


@tool(
    description=(
        "Calcula um percentil de uma coluna numérica. "
        "Útil para perguntas como valor no top 10%, mediana, quartis ou P90."
    ),
    parameters={
        "type": "object",
        "properties": {
            "coluna": {"type": "string", "description": "Coluna numérica."},
            "percentil": {
                "type": "number",
                "description": "Percentil entre 0 e 100, por exemplo 25, 50, 75 ou 90.",
            },
        },
        "required": ["coluna", "percentil"],
    },
)
def calcular_percentil(coluna: str, percentil: float) -> dict:
    """Calcula percentil de uma coluna numérica."""
    df = state.require_loaded()
    if coluna not in df.columns:
        return {"erro": f"Coluna '{coluna}' não existe."}
    if not pd.api.types.is_numeric_dtype(df[coluna]):
        return {"erro": f"Coluna '{coluna}' não é numérica."}
    if percentil < 0 or percentil > 100:
        return {"erro": "Percentil deve estar entre 0 e 100."}

    valor = df[coluna].dropna().quantile(percentil / 100)
    return {
        "coluna": coluna,
        "percentil": percentil,
        "valor": round(float(valor), 3),
    }


@tool(
    description=(
        "Cria uma tabela de contingência entre duas colunas categóricas. "
        "Use para cruzar categorias, por exemplo cidade × categoria ou sexo × renda."
    ),
    parameters={
        "type": "object",
        "properties": {
            "coluna_a": {"type": "string"},
            "coluna_b": {"type": "string"},
            "normalizar": {
                "type": "boolean",
                "description": "Se true, retorna proporções em vez de contagens.",
            },
        },
        "required": ["coluna_a", "coluna_b"],
    },
)
def tabela_contingencia(coluna_a: str, coluna_b: str, normalizar: bool = False) -> dict:
    """Cruzamento entre duas colunas categóricas."""
    df = state.require_loaded()
    for col in (coluna_a, coluna_b):
        if col not in df.columns:
            return {"erro": f"Coluna '{col}' não existe."}

    tabela = pd.crosstab(df[coluna_a], df[coluna_b], normalize="index" if normalizar else False)
    if normalizar:
        tabela = tabela.round(4)

    return {
        "coluna_a": coluna_a,
        "coluna_b": coluna_b,
        "normalizar": normalizar,
        "tabela": {
            str(idx): {str(col): float(valor) if normalizar else int(valor)
                       for col, valor in row.items()}
            for idx, row in tabela.iterrows()
        },
    }
