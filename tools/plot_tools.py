"""
Ferramenta de visualização: gera gráficos e salva como imagem.

As imagens são salvas em disco e a tool retorna apenas o caminho do arquivo gerado.
"""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .base import tool, state
from config import OUTPUTS_DIR


TIPOS_VALIDOS = {
    "hist",
    "histograma",
    "boxplot",
    "scatter",
    "barplot",
    "linha",
    "pizza",
    "pie",
}

AGREGACOES_VALIDAS = {
    "contagem",
    "count",
    "media",
    "média",
    "mean",
    "soma",
    "sum",
    "min",
    "max",
    "mediana",
    "median",
}


def _normalizar_texto(txt: str) -> str:
    return (
        str(txt)
        .strip()
        .lower()
        .replace("á", "a")
        .replace("à", "a")
        .replace("ã", "a")
        .replace("â", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("õ", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )


def _normalizar_agregacao(agregacao: str | None) -> str:
    if not agregacao:
        return "contagem"

    agg = _normalizar_texto(agregacao)

    if agg in {"contagem", "count", "quantidade", "frequencia", "frequência"}:
        return "contagem"
    if agg in {"media", "mean", "avg"}:
        return "media"
    if agg in {"soma", "sum", "total"}:
        return "soma"
    if agg in {"min", "minimo", "mínimo"}:
        return "min"
    if agg in {"max", "maximo", "máximo"}:
        return "max"
    if agg in {"mediana", "median"}:
        return "mediana"

    return "contagem"


def _formatar_valor(valor) -> str:
    try:
        valor_float = float(valor)
    except Exception:
        return str(valor)

    if valor_float.is_integer():
        return f"{valor_float:,.0f}".replace(",", ".")

    return f"{valor_float:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _gerar_nome_arquivo(
    tipo: str,
    colunas: list[str],
    agregacao: str | None = None,
    filtro_coluna: str | None = None,
    filtro_valor: str | None = None,
) -> Path:
    partes = [tipo]

    if agregacao:
        partes.append(str(agregacao))

    partes.extend(colunas)

    if filtro_coluna and filtro_valor:
        partes.extend([filtro_coluna, str(filtro_valor)])

    nome = "_".join(partes)
    nome = re.sub(r"[^a-zA-Z0-9_\-]+", "_", nome)
    nome = nome.strip("_")

    return OUTPUTS_DIR / f"plot_{nome}.png"


def _aplicar_filtro(
    df: pd.DataFrame,
    filtro_coluna: str | None,
    filtro_valor: str | None,
) -> pd.DataFrame:
    if not filtro_coluna or filtro_valor is None:
        return df

    if filtro_coluna not in df.columns:
        raise ValueError(f"Coluna de filtro '{filtro_coluna}' não existe.")

    serie = df[filtro_coluna]

    if pd.api.types.is_numeric_dtype(serie):
        try:
            valor_num = float(filtro_valor)
            return df[serie == valor_num]
        except ValueError:
            pass

    valor_norm = _normalizar_texto(filtro_valor)

    return df[serie.astype(str).map(_normalizar_texto) == valor_norm]


def _validar_numerica(df: pd.DataFrame, coluna: str) -> None:
    if not pd.api.types.is_numeric_dtype(df[coluna]):
        raise ValueError(f"A coluna '{coluna}' precisa ser numérica para este gráfico.")


def _adicionar_rotulos_barh(ax, serie: pd.Series) -> None:
    if serie.empty:
        return

    maior_valor = float(serie.max()) if len(serie) else 0
    deslocamento = maior_valor * 0.01 if maior_valor != 0 else 0.1

    for i, valor in enumerate(serie):
        ax.text(
            float(valor) + deslocamento,
            i,
            _formatar_valor(valor),
            va="center",
            fontsize=8,
        )

    if maior_valor > 0:
        ax.set_xlim(0, maior_valor * 1.18)


def _calcular_serie_categorica(
    df_plot: pd.DataFrame,
    colunas: list[str],
    agregacao_norm: str,
) -> tuple[pd.Series, str, str, dict]:
    """
    Retorna a série agregada, rótulo do eixo, título padrão e resultado em dict.
    Usado por barplot e pizza.
    """
    if len(colunas) == 1:
        coluna_categoria = colunas[0]

        serie = (
            df_plot[coluna_categoria]
            .value_counts(dropna=False)
            .sort_values(ascending=False)
            .head(20)
        )

        eixo = "Contagem"
        titulo_padrao = f"Contagem por {coluna_categoria}"
        resultado = serie.to_dict()

        return serie, eixo, titulo_padrao, resultado

    if len(colunas) != 2:
        raise ValueError(
            "Use 1 coluna categórica para contagem ou 2 colunas "
            "[categoria, numérica] para agregação."
        )

    coluna_categoria = colunas[0]
    coluna_numerica = colunas[1]

    _validar_numerica(df_plot, coluna_numerica)

    if agregacao_norm == "contagem":
        serie = df_plot.groupby(coluna_categoria)[coluna_numerica].count()
        eixo = "Contagem"
        titulo_padrao = f"Contagem de {coluna_numerica} por {coluna_categoria}"

    elif agregacao_norm == "media":
        serie = df_plot.groupby(coluna_categoria)[coluna_numerica].mean()
        eixo = f"Média de {coluna_numerica}"
        titulo_padrao = f"Média de {coluna_numerica} por {coluna_categoria}"

    elif agregacao_norm == "soma":
        serie = df_plot.groupby(coluna_categoria)[coluna_numerica].sum()
        eixo = f"Soma de {coluna_numerica}"
        titulo_padrao = f"Soma de {coluna_numerica} por {coluna_categoria}"

    elif agregacao_norm == "min":
        serie = df_plot.groupby(coluna_categoria)[coluna_numerica].min()
        eixo = f"Mínimo de {coluna_numerica}"
        titulo_padrao = f"Mínimo de {coluna_numerica} por {coluna_categoria}"

    elif agregacao_norm == "max":
        serie = df_plot.groupby(coluna_categoria)[coluna_numerica].max()
        eixo = f"Máximo de {coluna_numerica}"
        titulo_padrao = f"Máximo de {coluna_numerica} por {coluna_categoria}"

    elif agregacao_norm == "mediana":
        serie = df_plot.groupby(coluna_categoria)[coluna_numerica].median()
        eixo = f"Mediana de {coluna_numerica}"
        titulo_padrao = f"Mediana de {coluna_numerica} por {coluna_categoria}"

    else:
        raise ValueError(
            "Agregação inválida. Use: contagem, media, soma, min, max ou mediana."
        )

    serie = serie.sort_values(ascending=False).head(20)
    resultado = serie.round(4).to_dict()

    return serie, eixo, titulo_padrao, resultado


@tool(
    description=(
        "Gera um gráfico a partir do dataset e salva como PNG.\n\n"
        "Tipos disponíveis:\n"
        "  - 'hist' / 'histograma': histograma de UMA coluna numérica.\n"
        "  - 'boxplot': boxplot de UMA coluna numérica ou por categoria.\n"
        "  - 'scatter': dispersão entre DUAS colunas numéricas.\n"
        "  - 'barplot': gráfico de barras. Pode fazer contagem de uma coluna categórica "
        "ou agregação de uma coluna numérica por categoria.\n"
        "  - 'linha': gráfico de linha entre DUAS colunas, com eixo Y numérico.\n"
        "  - 'pizza' / 'pie': gráfico de pizza para participação por categoria.\n\n"
        "Para barplot e pizza:\n"
        "  - Se enviar 1 coluna, será feita contagem por categoria.\n"
        "  - Se enviar 2 colunas, use [categoria, valor_numerico]. "
        "A agregação pode ser: contagem, media, soma, min, max ou mediana.\n\n"
        "Exemplos:\n"
        "  gerar_grafico(tipo='barplot', colunas=['categoria']) -> contagem por categoria.\n"
        "  gerar_grafico(tipo='barplot', colunas=['categoria', 'valor_total'], agregacao='media') "
        "-> média de valor_total por categoria.\n"
        "  gerar_grafico(tipo='pizza', colunas=['categoria', 'valor_total'], agregacao='soma') "
        "-> participação de valor_total por categoria.\n"
        "  gerar_grafico(tipo='scatter', colunas=['quantidade', 'valor_total']) "
        "-> dispersão entre duas colunas numéricas.\n\n"
        "Retorna o caminho do arquivo PNG gerado."
    ),
    parameters={
        "type": "object",
        "properties": {
            "tipo": {
                "type": "string",
                "enum": list(TIPOS_VALIDOS),
                "description": "Tipo do gráfico.",
            },
            "colunas": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Colunas envolvidas. Para barplot ou pizza com agregação, use "
                    "[coluna_categorica, coluna_numerica]."
                ),
            },
            "titulo": {
                "type": "string",
                "description": "Título do gráfico.",
            },
            "agregacao": {
                "type": "string",
                "enum": list(AGREGACOES_VALIDAS),
                "description": (
                    "Agregação usada no barplot ou pizza quando houver coluna numérica: "
                    "contagem, media, soma, min, max ou mediana."
                ),
            },
            "filtro_coluna": {
                "type": "string",
                "description": "Coluna opcional usada para filtrar antes de gerar o gráfico.",
            },
            "filtro_valor": {
                "type": "string",
                "description": "Valor opcional do filtro.",
            },
        },
        "required": ["tipo", "colunas"],
    },
)
def gerar_grafico(
    tipo: str,
    colunas: list[str],
    titulo: str = "",
    agregacao: str | None = None,
    filtro_coluna: str | None = None,
    filtro_valor: str | None = None,
) -> dict:
    """
    Gera gráfico e retorna caminho do arquivo.

    Melhorias:
    - barplot com 1 coluna: contagem por categoria.
    - barplot com 2 colunas: agregação numérica por categoria.
    - barras coloridas e com valor exato impresso.
    - pizza para participação por categoria.
    - scatter validado apenas para colunas numéricas.
    - linha ordenada pelo eixo X.
    - filtros simples podem ser aplicados antes de gerar o gráfico.
    """
    df = state.require_loaded()

    tipo = _normalizar_texto(tipo)
    if tipo == "pie":
        tipo = "pizza"

    agregacao_norm = _normalizar_agregacao(agregacao)

    if tipo not in TIPOS_VALIDOS:
        return {"erro": f"Tipo '{tipo}' inválido. Use um de {sorted(TIPOS_VALIDOS)}."}

    if not colunas or not isinstance(colunas, list):
        return {"erro": "Informe uma lista de colunas."}

    for col in colunas:
        if col not in df.columns:
            return {
                "erro": f"Coluna '{col}' não existe.",
                "colunas_disponiveis": list(df.columns),
            }

    if filtro_coluna and filtro_coluna not in df.columns:
        return {
            "erro": f"Coluna de filtro '{filtro_coluna}' não existe.",
            "colunas_disponiveis": list(df.columns),
        }

    fig = None

    try:
        df_plot = _aplicar_filtro(df, filtro_coluna, filtro_valor)

        if df_plot.empty:
            return {
                "erro": "Nenhum registro encontrado após aplicar o filtro.",
                "filtro": f"{filtro_coluna}={filtro_valor}",
            }

        fig, ax = plt.subplots(figsize=(9, 5.5))

        resultado = None

        if tipo in {"hist", "histograma"}:
            if len(colunas) != 1:
                return {"erro": "Histograma requer exatamente 1 coluna numérica."}

            coluna = colunas[0]
            _validar_numerica(df_plot, coluna)

            df_plot[coluna].dropna().plot(
                kind="hist",
                bins=30,
                ax=ax,
                edgecolor="black",
                alpha=0.8,
            )

            ax.set_xlabel(coluna)
            ax.set_ylabel("Frequência")
            ax.set_title(titulo or f"Distribuição de {coluna}")

        elif tipo == "boxplot":
            if len(colunas) == 1:
                coluna = colunas[0]
                _validar_numerica(df_plot, coluna)

                df_plot[coluna].dropna().plot(
                    kind="box",
                    ax=ax,
                )

                ax.set_ylabel(coluna)
                ax.set_title(titulo or f"Boxplot de {coluna}")

            elif len(colunas) == 2:
                coluna_categoria = colunas[0]
                coluna_numerica = colunas[1]

                _validar_numerica(df_plot, coluna_numerica)

                df_plot.boxplot(
                    column=coluna_numerica,
                    by=coluna_categoria,
                    ax=ax,
                    grid=False,
                )

                ax.set_xlabel(coluna_categoria)
                ax.set_ylabel(coluna_numerica)
                ax.set_title(titulo or f"{coluna_numerica} por {coluna_categoria}")
                fig.suptitle("")

            else:
                return {
                    "erro": "Boxplot aceita 1 coluna numérica ou 2 colunas: [categoria, numérica]."
                }

        elif tipo == "scatter":
            if len(colunas) != 2:
                return {"erro": "Scatter requer exatamente 2 colunas numéricas."}

            x, y = colunas
            _validar_numerica(df_plot, x)
            _validar_numerica(df_plot, y)

            df_plot.plot.scatter(
                x=x,
                y=y,
                ax=ax,
                alpha=0.65,
                s=45,
            )

            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(titulo or f"Relação entre {x} e {y}")

        elif tipo == "barplot":
            serie, xlabel, titulo_padrao, resultado = _calcular_serie_categorica(
                df_plot=df_plot,
                colunas=colunas,
                agregacao_norm=agregacao_norm,
            )

            serie_plot = serie.sort_values(ascending=True)
            cores = plt.cm.tab20(range(len(serie_plot)))

            serie_plot.plot(
                kind="barh",
                ax=ax,
                color=cores,
            )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(colunas[0])
            ax.set_title(titulo or titulo_padrao)

            _adicionar_rotulos_barh(ax, serie_plot)

            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        elif tipo == "pizza":
            serie, _, titulo_padrao, resultado = _calcular_serie_categorica(
                df_plot=df_plot,
                colunas=colunas,
                agregacao_norm=agregacao_norm,
            )

            # Pizza com muitas categorias fica ruim; limita aos 8 maiores.
            serie_plot = serie.sort_values(ascending=False).head(8)

            if len(serie) > 8:
                outros = serie.sort_values(ascending=False).iloc[8:].sum()
                if outros > 0:
                    serie_plot.loc["Outros"] = outros

            cores = plt.cm.tab20(range(len(serie_plot)))

            serie_plot.plot(
                kind="pie",
                ax=ax,
                autopct="%1.1f%%",
                startangle=90,
                colors=cores,
                ylabel="",
            )

            ax.set_title(titulo or titulo_padrao.replace("Contagem", "Participação"))

        elif tipo == "linha":
            if len(colunas) != 2:
                return {"erro": "Linha requer exatamente 2 colunas: [x, y]."}

            x, y = colunas
            _validar_numerica(df_plot, y)

            df_linha = df_plot[[x, y]].dropna().sort_values(by=x)

            df_linha.plot(
                x=x,
                y=y,
                ax=ax,
                kind="line",
                legend=False,
                marker="o",
                linewidth=2,
            )

            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(titulo or f"{y} por {x}")

        if tipo == "barplot":
            ax.grid(axis="x", alpha=0.25)
        elif tipo != "pizza":
            ax.grid(axis="y", alpha=0.25)

        if tipo not in {"barplot", "pizza"}:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        fig.tight_layout()

        caminho = _gerar_nome_arquivo(
            tipo=tipo,
            colunas=colunas,
            agregacao=agregacao_norm if tipo in {"barplot", "pizza"} else None,
            filtro_coluna=filtro_coluna,
            filtro_valor=filtro_valor,
        )

        fig.savefig(caminho, dpi=120)
        plt.close(fig)

        retorno = {
            "tipo": tipo,
            "colunas": colunas,
            "arquivo_gerado": str(caminho),
            "mensagem": f"Gráfico salvo em {caminho.name}",
        }

        if tipo in {"barplot", "pizza"}:
            retorno["agregacao"] = agregacao_norm
            retorno["resultado_plotado"] = resultado

        if filtro_coluna and filtro_valor is not None:
            retorno["filtro"] = f"{filtro_coluna}={filtro_valor}"
            retorno["linhas_apos_filtro"] = int(len(df_plot))

        return retorno

    except Exception as e:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass

        return {"erro": f"Erro ao gerar gráfico: {e}"}