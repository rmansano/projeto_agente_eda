"""
Ferramentas de inspeção do dataset.

Estas tools respondem a perguntas do tipo "o que tem nesse CSV?"
e geralmente são as PRIMEIRAS a serem chamadas pelo agente quando
ele recebe uma pergunta nova.
"""

from .base import tool, state


# ============================================================
# listar_colunas
# ============================================================

@tool(
    description=(
        "Retorna a lista de colunas do dataset com seus tipos. "
        "Use esta tool SEMPRE que precisar saber quais colunas existem "
        "antes de operar sobre elas."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
def listar_colunas() -> dict:
    """
    Lista colunas e tipos do dataset carregado.

    Returns:
        dict com a chave 'colunas', cujo valor é uma lista de
        dicionários {nome, tipo}.
    """
    df = state.require_loaded()
    return {
        "colunas": [
            {"nome": col, "tipo": str(df[col].dtype)}
            for col in df.columns
        ],
        "total_linhas": len(df),
        "total_colunas": len(df.columns),
    }


# ============================================================
# descrever_dados
# ============================================================

@tool(
    description=(
        "Retorna estatísticas descritivas do dataset (equivalente a df.describe()), "
        "incluindo tanto colunas numéricas quanto categóricas."
    ),
    parameters={
        "type": "object",
        "properties": {
            "colunas": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Lista de colunas a descrever. Se vazia ou omitida, "
                    "descreve todas as colunas."
                ),
            },
        },
        "required": [],
    },
)
def descrever_dados(colunas: list[str] | None = None) -> dict:
    """
    Retorna estatísticas descritivas.

    Args:
        colunas: subconjunto de colunas. Se None, usa todas.

    Returns:
        dict com 'numericas' e 'categoricas' como chaves.
    """
    df = state.require_loaded()

    if colunas:
        # Valida que todas as colunas existem
        invalidas = [c for c in colunas if c not in df.columns]
        if invalidas:
            return {"erro": f"Colunas inexistentes: {invalidas}"}
        df = df[colunas]

    # Separa numéricas de categóricas
    num_df = df.select_dtypes(include="number")
    cat_df = df.select_dtypes(exclude="number")

    resultado = {}
    if not num_df.empty:
        resultado["numericas"] = num_df.describe().round(3).to_dict()
    if not cat_df.empty:
        resultado["categoricas"] = {
            col: {
                "valores_unicos": int(cat_df[col].nunique()),
                "mais_frequente": str(cat_df[col].mode().iloc[0])
                                  if not cat_df[col].mode().empty else None,
                "frequencia_top": int(cat_df[col].value_counts().iloc[0])
                                  if len(cat_df[col]) > 0 else 0,
            }
            for col in cat_df.columns
        }
    return resultado


# ============================================================
# contar_valores
# ============================================================

@tool(
    description=(
        "Retorna a distribuição de valores de uma coluna específica "
        "(equivalente a value_counts). Útil para entender categorias."
    ),
    parameters={
        "type": "object",
        "properties": {
            "coluna": {
                "type": "string",
                "description": "Nome da coluna a analisar.",
            },
            "top_n": {
                "type": "integer",
                "description": "Quantos valores mais frequentes retornar (default: 10).",
            },
        },
        "required": ["coluna"],
    },
)
def contar_valores(coluna: str, top_n: int = 10) -> dict:
    """Distribuição de valores de uma coluna."""
    df = state.require_loaded()

    if coluna not in df.columns:
        return {"erro": f"Coluna '{coluna}' não existe no dataset."}

    contagem = df[coluna].value_counts().head(top_n)
    return {
        "coluna": coluna,
        "total_valores_unicos": int(df[coluna].nunique()),
        "distribuicao": {str(k): int(v) for k, v in contagem.items()},
        "valores_nulos": int(df[coluna].isna().sum()),
    }
