"""
Ferramentas de filtragem e agregação.

Estas tools transformam o dataset (recortes, grupos) e produzem
estatísticas resumidas. Geralmente vêm DEPOIS de uma chamada a
listar_colunas, quando o agente já sabe o que filtrar.
"""

import pandas as pd
from .base import tool, state


# ============================================================
# filtrar
# ============================================================

@tool(
    description=(
        "Filtra o dataset com uma expressão de consulta (sintaxe pandas .query) "
        "e retorna estatísticas do subconjunto resultante.\n\n"
        "Exemplos de expressão válida:\n"
        "  - 'idade > 30'\n"
        "  - \"sexo == 'F' and renda > 5000\"\n"
        "  - 'pais in [\"Brasil\", \"Argentina\"]'\n\n"
        "Atenção: nomes de colunas com espaço devem usar crase no pandas query."
    ),
    parameters={
        "type": "object",
        "properties": {
            "condicao": {
                "type": "string",
                "description": "Expressão de filtro no formato pandas .query()",
            },
        },
        "required": ["condicao"],
    },
)
def filtrar(condicao: str) -> dict:
    """
    Filtra e retorna estatísticas do subconjunto.

    Note: a expressão é avaliada por df.query() — segura porque o pandas
    não permite execução de código arbitrário neste contexto.
    """
    df = state.require_loaded()

    try:
        filtrado = df.query(condicao)
    except Exception as e:
        return {
            "erro": f"Expressão inválida: {e}",
            "dica": "Verifique os nomes das colunas e a sintaxe pandas query.",
        }

    if len(filtrado) == 0:
        return {
            "condicao": condicao,
            "linhas_resultantes": 0,
            "aviso": "Nenhuma linha satisfaz a condição.",
        }

    # Estatísticas numéricas do recorte
    num_cols = filtrado.select_dtypes(include="number").columns.tolist()
    estatisticas = {}
    for col in num_cols:
        estatisticas[col] = {
            "media": round(float(filtrado[col].mean()), 3),
            "mediana": round(float(filtrado[col].median()), 3),
            "min": round(float(filtrado[col].min()), 3),
            "max": round(float(filtrado[col].max()), 3),
        }

    return {
        "condicao": condicao,
        "linhas_resultantes": len(filtrado),
        "porcentagem_do_total": round(len(filtrado) / len(df) * 100, 2),
        "estatisticas": estatisticas,
    }


# ============================================================
# agrupar_e_agregar
# ============================================================

# Funções de agregação aceitas
FUNCOES_VALIDAS = {"mean", "median", "sum", "min", "max", "count", "std"}


@tool(
    description=(
        "Agrupa o dataset por uma coluna e aplica uma função de agregação "
        "sobre outra coluna. Equivalente a df.groupby(grupo)[coluna].agg(funcao).\n\n"
        "Funções válidas: mean, median, sum, min, max, count, std."
    ),
    parameters={
        "type": "object",
        "properties": {
            "grupo": {
                "type": "string",
                "description": "Coluna pela qual agrupar (geralmente categórica).",
            },
            "coluna": {
                "type": "string",
                "description": "Coluna sobre a qual aplicar a agregação (geralmente numérica).",
            },
            "funcao": {
                "type": "string",
                "enum": list(FUNCOES_VALIDAS),
                "description": "Função de agregação.",
            },
        },
        "required": ["grupo", "coluna", "funcao"],
    },
)
def agrupar_e_agregar(grupo: str, coluna: str, funcao: str) -> dict:
    """Groupby + agg."""
    df = state.require_loaded()

    # Validações
    if grupo not in df.columns:
        return {"erro": f"Coluna de grupo '{grupo}' não existe."}
    if coluna not in df.columns:
        return {"erro": f"Coluna '{coluna}' não existe."}
    if funcao not in FUNCOES_VALIDAS:
        return {"erro": f"Função '{funcao}' inválida. Use uma de {FUNCOES_VALIDAS}."}

    # 'count' funciona em qualquer tipo; as demais exigem numérico
    if funcao != "count" and not pd.api.types.is_numeric_dtype(df[coluna]):
        return {
            "erro": (
                f"Função '{funcao}' requer coluna numérica, "
                f"mas '{coluna}' é {df[coluna].dtype}."
            )
        }

    resultado = df.groupby(grupo)[coluna].agg(funcao)

    return {
        "grupo": grupo,
        "coluna": coluna,
        "funcao": funcao,
        "resultados": {
            str(k): round(float(v), 3) if pd.notna(v) else None
            for k, v in resultado.items()
        },
    }
