"""
Ferramenta de visualização: gera gráficos e salva como imagem.

Decisão de design: as imagens são salvas em disco e a tool retorna
APENAS o caminho do arquivo gerado. Não tentamos passar imagem ao LLM
porque isso aumenta muito o custo de tokens e não é necessário —
basta o agente avisar ao usuário onde está o gráfico.
"""

from datetime import datetime
from pathlib import Path
import matplotlib
# Backend não-interativo: importante quando rodando via CLI/CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .base import tool, state
from config import OUTPUTS_DIR


TIPOS_VALIDOS = {"hist", "histograma", "boxplot", "scatter", "barplot", "linha"}


def _gerar_nome_arquivo(tipo: str, colunas: list[str]) -> Path:
    """
    Gera nome determinístico para evitar vários gráficos iguais.
    Se o mesmo tipo e as mesmas colunas forem usados novamente,
    o arquivo anterior é sobrescrito.
    """
    nome_colunas = "_".join(colunas)
    nome_colunas = (
        nome_colunas
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    return OUTPUTS_DIR / f"plot_{tipo}_{nome_colunas}.png"


@tool(
    description=(
        "Gera um gráfico a partir do dataset e salva como PNG.\n\n"
        "Tipos disponíveis:\n"
        "  - 'hist' / 'histograma': histograma de UMA coluna numérica\n"
        "  - 'boxplot': boxplot de UMA coluna (opcionalmente por categoria)\n"
        "  - 'scatter': dispersão entre DUAS colunas numéricas\n"
        "  - 'barplot': barras com contagem de UMA coluna categórica\n"
        "  - 'linha': gráfico de linha entre DUAS colunas (ex.: tempo × valor)\n\n"
        "Retorna o caminho do arquivo PNG gerado."
    ),
    parameters={
        "type": "object",
        "properties": {
            "tipo": {
                "type": "string",
                "enum": list(TIPOS_VALIDOS),
            },
            "colunas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Colunas envolvidas. Quantidade depende do tipo.",
            },
            "titulo": {
                "type": "string",
                "description": "Título do gráfico (opcional).",
            },
        },
        "required": ["tipo", "colunas"],
    },
)
def gerar_grafico(tipo: str, colunas: list[str], titulo: str = "") -> dict:
    """Gera gráfico e retorna caminho do arquivo."""
    df = state.require_loaded()

    # Valida tipo
    if tipo not in TIPOS_VALIDOS:
        return {"erro": f"Tipo '{tipo}' inválido. Use um de {TIPOS_VALIDOS}."}

    # Valida colunas
    for col in colunas:
        if col not in df.columns:
            return {"erro": f"Coluna '{col}' não existe."}

    # Cria figura
    fig, ax = plt.subplots(figsize=(8, 5))

    try:
        if tipo in {"hist", "histograma"}:
            if len(colunas) != 1:
                return {"erro": "Histograma requer exatamente 1 coluna."}
            df[colunas[0]].plot(kind="hist", bins=30, ax=ax, edgecolor="black")
            ax.set_xlabel(colunas[0])
            ax.set_ylabel("Frequência")

        elif tipo == "boxplot":
            if len(colunas) == 1:
                df[colunas[0]].plot(kind="box", ax=ax)
            elif len(colunas) == 2:
                # 1 categórica + 1 numérica
                df.boxplot(column=colunas[1], by=colunas[0], ax=ax)
            else:
                return {"erro": "Boxplot aceita 1 coluna (uma série) ou 2 (categoria × numérica)."}

        elif tipo == "scatter":
            if len(colunas) != 2:
                return {"erro": "Scatter requer exatamente 2 colunas."}
            df.plot.scatter(x=colunas[0], y=colunas[1], ax=ax, alpha=0.5)

        elif tipo == "barplot":
            if len(colunas) != 1:
                return {"erro": "Barplot requer exatamente 1 coluna."}
            df[colunas[0]].value_counts().head(20).plot(kind="bar", ax=ax)
            ax.set_xlabel(colunas[0])
            ax.set_ylabel("Contagem")

        elif tipo == "linha":
            if len(colunas) != 2:
                return {"erro": "Linha requer 2 colunas (x e y)."}
            df.plot(x=colunas[0], y=colunas[1], ax=ax, kind="line")

        ax.set_title(titulo or f"{tipo}: {', '.join(colunas)}")
        fig.tight_layout()

        caminho = _gerar_nome_arquivo(tipo, colunas)
        fig.savefig(caminho, dpi=120)
        plt.close(fig)

        return {
            "tipo": tipo,
            "colunas": colunas,
            "arquivo_gerado": str(caminho),
            "mensagem": f"Gráfico salvo em {caminho.name}",
        }

    except Exception as e:
        plt.close(fig)
        return {"erro": f"Erro ao gerar gráfico: {e}"}
