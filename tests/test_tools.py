"""
Testes unitários das tools.

Estes testes não dependem do LLM — testam APENAS a lógica em pandas.
Rodar com:
    pytest tests/

TODO (alunos):
  - Adicionar testes para as tools que vocês criarem.
  - Adicionar testes de casos de erro (coluna inexistente, etc).
"""

import pandas as pd
import pytest

from tools import state
from tools.inspect_tools import listar_colunas, descrever_dados, contar_valores
from tools.filter_tools import filtrar, agrupar_e_agregar
from tools.stats_tools import correlacao, detectar_outliers


# ============================================================
# Fixture: dataset sintético usado por todos os testes
# ============================================================

@pytest.fixture(autouse=True)
def carregar_dataset_sintetico():
    """Carrega um DataFrame sintético antes de cada teste."""
    df = pd.DataFrame({
        "idade": [25, 30, 35, 40, 45, 50, 100],   # 100 é outlier
        "salario": [3000, 4500, 6000, 7500, 9000, 11000, 12500],
        "genero": ["F", "M", "F", "M", "F", "M", "F"],
        "cidade": ["SP", "RJ", "SP", "MG", "RJ", "SP", "BA"],
    })
    state.df = df
    state.path = "<fixture>"
    yield
    state.df = None


# ============================================================
# Testes de inspect_tools
# ============================================================

def test_listar_colunas_retorna_todas():
    resultado = listar_colunas()
    nomes = [c["nome"] for c in resultado["colunas"]]
    assert nomes == ["idade", "salario", "genero", "cidade"]
    assert resultado["total_linhas"] == 7
    assert resultado["total_colunas"] == 4


def test_descrever_dados_separa_numericas_de_categoricas():
    resultado = descrever_dados()
    assert "numericas" in resultado
    assert "categoricas" in resultado
    assert "idade" in resultado["numericas"]
    assert "genero" in resultado["categoricas"]


def test_descrever_dados_coluna_invalida_retorna_erro():
    resultado = descrever_dados(colunas=["nao_existe"])
    assert "erro" in resultado


def test_contar_valores_basico():
    resultado = contar_valores("genero")
    assert resultado["coluna"] == "genero"
    assert resultado["total_valores_unicos"] == 2
    assert resultado["distribuicao"]["F"] == 4
    assert resultado["distribuicao"]["M"] == 3


def test_contar_valores_coluna_invalida():
    resultado = contar_valores("inexistente")
    assert "erro" in resultado


# ============================================================
# Testes de filter_tools
# ============================================================

def test_filtrar_basico():
    resultado = filtrar("idade > 35")
    assert resultado["linhas_resultantes"] == 4  # 40, 45, 50, 100


def test_filtrar_expressao_invalida():
    resultado = filtrar("coluna_inexistente > 0")
    assert "erro" in resultado


def test_agrupar_e_agregar_media_por_genero():
    resultado = agrupar_e_agregar(grupo="genero", coluna="salario", funcao="mean")
    assert "F" in resultado["resultados"]
    assert "M" in resultado["resultados"]
    # F: (3000+6000+9000+12500)/4 = 7625
    assert resultado["resultados"]["F"] == pytest.approx(7625.0, abs=0.1)


def test_agrupar_e_agregar_funcao_invalida():
    resultado = agrupar_e_agregar(grupo="genero", coluna="salario", funcao="xyz")
    assert "erro" in resultado


def test_agrupar_e_agregar_coluna_nao_numerica():
    resultado = agrupar_e_agregar(grupo="genero", coluna="cidade", funcao="mean")
    assert "erro" in resultado


# ============================================================
# Testes de stats_tools
# ============================================================

def test_correlacao_idade_salario():
    # Idade e salário são fortemente correlacionados nesse dataset
    resultado = correlacao("idade", "salario")
    assert "correlacao" in resultado
    assert resultado["correlacao"] > 0.8  # esperamos forte positiva


def test_correlacao_coluna_categorica_retorna_erro():
    resultado = correlacao("idade", "genero")
    assert "erro" in resultado


def test_detectar_outliers_iqr_identifica_o_100():
    resultado = detectar_outliers("idade", metodo="iqr")
    assert resultado["total_outliers"] >= 1
    assert 100.0 in resultado["exemplos"]


def test_detectar_outliers_metodo_invalido():
    resultado = detectar_outliers("idade", metodo="foo")
    assert "erro" in resultado

# ============================================================
# Testes das tools extras
# ============================================================

from tools.extra_tools import contar_nulos, calcular_percentil, tabela_contingencia


def test_detectar_outliers_zscore_funciona():
    resultado = detectar_outliers("idade", metodo="zscore")
    assert "erro" not in resultado
    assert resultado["metodo"] == "zscore"
    assert "total_outliers" in resultado
    assert "limite_z" in resultado


def test_contar_nulos_retorna_por_coluna():
    resultado = contar_nulos()
    assert resultado["total_linhas"] == 7
    assert "idade" in resultado["por_coluna"]
    assert resultado["por_coluna"]["idade"]["nulos"] == 0


def test_calcular_percentil_mediana():
    resultado = calcular_percentil("idade", 50)
    assert "erro" not in resultado
    assert resultado["valor"] == pytest.approx(40.0, abs=0.1)


def test_calcular_percentil_coluna_categorica_retorna_erro():
    resultado = calcular_percentil("genero", 90)
    assert "erro" in resultado


def test_tabela_contingencia_basica():
    resultado = tabela_contingencia("genero", "cidade")
    assert "erro" not in resultado
    assert "F" in resultado["tabela"]
    assert "SP" in resultado["tabela"]["F"]
