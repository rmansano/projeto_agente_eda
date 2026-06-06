from .base import tool, state


@tool(
    description="Detecta correlações fortes entre colunas numéricas do dataset.",
    parameters={
        "type": "object",
        "properties": {
            "limite": {
                "type": "number",
                "description": "Valor mínimo absoluto da correlação. Ex: 0.7",
            }
        },
        "required": [],
    },
)
def detectar_correlacoes(limite: float = 0.7) -> dict:
    df = state.require_loaded()
    num = df.select_dtypes(include="number")

    if num.shape[1] < 2:
        return {"erro": "Não há colunas numéricas suficientes."}

    corr = num.corr(numeric_only=True)
    resultados = []

    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            valor = round(float(corr.loc[cols[i], cols[j]]), 4)
            if abs(valor) >= limite:
                resultados.append({
                    "coluna_a": cols[i],
                    "coluna_b": cols[j],
                    "correlacao": valor,
                    "forca": "forte" if abs(valor) >= 0.7 else "moderada",
                    "direcao": "positiva" if valor > 0 else "negativa",
                })

    return {
        "limite": limite,
        "total_correlacoes_encontradas": len(resultados),
        "correlacoes": resultados,
    }


@tool(
    description="Faz um diagnóstico geral do dataset: nulos, duplicados, tipos, constantes e outliers numéricos.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
def diagnosticar_dataset() -> dict:
    df = state.require_loaded()

    nulos = df.isna().sum()
    duplicados = int(df.duplicated().sum())

    colunas_constantes = [
        col for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]

    outliers = {}
    for col in df.select_dtypes(include="number").columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lim_inf = q1 - 1.5 * iqr
        lim_sup = q3 + 1.5 * iqr

        total = int(((df[col] < lim_inf) | (df[col] > lim_sup)).sum())

        if total > 0:
            outliers[col] = {
                "total": total,
                "porcentagem": round((total / len(df)) * 100, 2),
                "limite_inferior": round(float(lim_inf), 3),
                "limite_superior": round(float(lim_sup), 3),
            }

    return {
        "linhas": int(df.shape[0]),
        "colunas": int(df.shape[1]),
        "duplicados": duplicados,
        "total_nulos": int(nulos.sum()),
        "nulos_por_coluna": {
            col: int(valor)
            for col, valor in nulos.items()
            if int(valor) > 0
        },
        "colunas_constantes": colunas_constantes,
        "outliers_detectados": outliers,
        "tipos": {
            col: str(tipo)
            for col, tipo in df.dtypes.items()
        },
    }


@tool(
    description="Gera insights automáticos simples sobre o dataset, destacando padrões relevantes.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
def gerar_insights() -> dict:
    df = state.require_loaded()

    insights = []

    # Nulos
    total_nulos = int(df.isna().sum().sum())
    if total_nulos == 0:
        insights.append("O dataset não possui valores nulos.")
    else:
        insights.append(f"O dataset possui {total_nulos} valores nulos.")

    # Duplicados
    duplicados = int(df.duplicated().sum())
    if duplicados == 0:
        insights.append("Não foram encontrados registros duplicados.")
    else:
        insights.append(f"Foram encontrados {duplicados} registros duplicados.")

    # Categóricas mais frequentes
    for col in df.select_dtypes(exclude="number").columns[:5]:
        moda = df[col].mode(dropna=True)
        if not moda.empty:
            valor = moda.iloc[0]
            qtd = int((df[col] == valor).sum())
            insights.append(
                f"Na coluna '{col}', o valor mais frequente é '{valor}' com {qtd} ocorrências."
            )

    # Numéricas: média, mínimo e máximo
    for col in df.select_dtypes(include="number").columns[:5]:
        media = round(float(df[col].mean()), 3)
        minimo = round(float(df[col].min()), 3)
        maximo = round(float(df[col].max()), 3)
        insights.append(
            f"A coluna '{col}' tem média {media}, mínimo {minimo} e máximo {maximo}."
        )

    # Correlação mais forte
    num = df.select_dtypes(include="number")
    if num.shape[1] >= 2:
        corr = num.corr(numeric_only=True).abs()
        pares = []

        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pares.append((cols[i], cols[j], float(corr.loc[cols[i], cols[j]])))

        if pares:
            a, b, valor = max(pares, key=lambda x: x[2])
            insights.append(
                f"A maior correlação numérica encontrada foi entre '{a}' e '{b}' ({round(valor, 4)})."
            )

    return {
        "total_insights": len(insights),
        "insights": insights,
    }