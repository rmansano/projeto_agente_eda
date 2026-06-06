
from __future__ import annotations
import unicodedata
from config import NUMERIC_TOLERANCE
import re
from dataclasses import dataclass

from config import (
    NUMERIC_TOLERANCE,
    INPUT_COST_PER_1M_TOKENS,
    OUTPUT_COST_PER_1M_TOKENS,
)

def normalizar_texto(texto: str) -> str:
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto

def extrair_numeros(texto: str) -> list[float]:
    if not texto:
        return []

    texto = str(texto).replace("R$", "").replace("%", "").replace("$", "")

    padrao = r"-?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|-?\d+[.,]\d+|-?\d+"
    encontrados = re.findall(padrao, texto)

    numeros = []

    for valor in encontrados:
        if "," in valor and "." in valor:
            if valor.rfind(",") > valor.rfind("."):
                valor = valor.replace(".", "").replace(",", ".")
            else:
                valor = valor.replace(",", "")
        elif "," in valor:
            valor = valor.replace(",", ".")
        elif "." in valor:
            partes = valor.split(".")
            if len(partes[-1]) == 3 and len(partes) > 1:
                valor = valor.replace(".", "")

        try:
            numeros.append(float(valor))
        except ValueError:
            pass

    return numeros

def numero_bate(valor_obtido: float, valor_esperado: float) -> bool:
    tolerancia = max(NUMERIC_TOLERANCE, abs(valor_esperado) * 0.01)

    # Comparação direta: 0.05 com 0.05
    if abs(valor_obtido - valor_esperado) <= tolerancia:
        return True

    # Comparação percentual: 5.0% com 0.05
    if abs((valor_obtido / 100) - valor_esperado) <= tolerancia:
        return True

    # Comparação inversa: 0.05 com 5.0%
    if abs(valor_obtido - (valor_esperado * 100)) <= max(NUMERIC_TOLERANCE, abs(valor_esperado * 100) * 0.01):
        return True

    return False

def comparar_numero(resposta: str, esperado: float) -> bool:
    resposta_lower = str(resposta).lower()

    # Caso especial: esperado é zero, mas o agente responde em linguagem natural
    # Ex.: "não possui nenhum valor nulo"
    if float(esperado) == 0:
        expressoes_zero = [
            "nenhum",
            "nenhuma",
            "não possui",
            "nao possui",
            "não há",
            "nao ha",
            "sem nulos",
            "sem valores nulos",
            "não existem",
            "nao existem",
            "não tem",
            "nao tem",
        ]

        if any(expr in resposta_lower for expr in expressoes_zero):
            return True

    numeros = extrair_numeros(resposta)

    return any(numero_bate(n, float(esperado)) for n in numeros)


def comparar_lista_strings(resposta: str, esperado: list[str]) -> bool:
    resposta_lower = str(resposta).lower()
    return all(str(item).lower() in resposta_lower for item in esperado)


def comparar_dict_numerico(resposta: str, esperado: dict) -> bool:
    resposta_norm = normalizar_texto(resposta)

    for chave, valor_esperado in esperado.items():
        chave_norm = normalizar_texto(chave)

        if chave_norm not in resposta_norm:
            return False

        idx = resposta_norm.find(chave_norm)

        # Pega um trecho perto da chave, porque o valor correto costuma estar na mesma linha/tabela
        trecho = resposta[max(0, idx - 80): idx + 180]
        numeros = extrair_numeros(trecho)

        if not numeros:
            return False

        if not any(numero_bate(n, float(valor_esperado)) for n in numeros):
            return False

    return True

PALAVRAS_RECUSA = {
    "ambígua", "ambigua", "não entendi", "nao entendi",
    "esclarecer", "esclareça", "esclareca",
    "não consigo", "nao consigo",
    "não posso", "nao posso",
    "inválida", "invalida",
    "não está clara", "nao esta clara",
    "não existe", "nao existe", "coluna inexistente",
    "fora do escopo", "não faz parte", "nao faz parte",
    "não é possível", "nao e possivel", "não é possível responder",
}


def comparar_categorica(resposta: str, esperado: str) -> bool:
    resposta_lower = str(resposta).lower()

    if esperado == "recusa":
        return any(palavra in resposta_lower for palavra in PALAVRAS_RECUSA)

    return str(esperado).lower() in resposta_lower


def avaliar_resposta(resposta: str, esperado, tipo_resposta: str) -> bool:
    if esperado is None:
        return False

    if tipo_resposta in ("numero_inteiro", "numero_float"):
        return comparar_numero(resposta, float(esperado))

    if tipo_resposta == "lista_strings":
        return comparar_lista_strings(resposta, esperado)

    if tipo_resposta == "dict_numerico":
        return comparar_dict_numerico(resposta, esperado)

    if tipo_resposta == "categorica":
        return comparar_categorica(resposta, esperado)

    raise ValueError(f"Tipo de resposta desconhecido: {tipo_resposta}")


def estimar_custo_usd(input_tokens: int, output_tokens: int) -> float:
    custo_input = (input_tokens / 1_000_000) * INPUT_COST_PER_1M_TOKENS
    custo_output = (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M_TOKENS
    return round(custo_input + custo_output, 6)


@dataclass
class BenchmarkSummary:
    total_perguntas: int
    acertos: int
    taxa_execucao_sucesso: float
    acuracia_geral: float
    acuracia_por_tipo: dict[str, float]
    tool_calls_media: float
    latencia_media: float
    input_tokens_total: int
    output_tokens_total: int
    custo_total_usd: float
    custo_medio_usd: float

    def imprimir(self):
        print("\n" + "=" * 60)
        print("RESUMO DO BENCHMARK")
        print("=" * 60)
        print(f"Total de perguntas:            {self.total_perguntas}")
        print(f"Acertos:                       {self.acertos}")
        print(f"Acurácia geral:                {self.acuracia_geral:.1%}")
        print(f"Taxa de execução bem-sucedida: {self.taxa_execucao_sucesso:.1%}")
        print()
        print("Acurácia por tipo de pergunta:")
        for tipo, acc in self.acuracia_por_tipo.items():
            print(f"  - {tipo:15s}: {acc:.1%}")
        print()
        print(f"Tool calls médias por pergunta: {self.tool_calls_media:.2f}")
        print(f"Latência média por pergunta:    {self.latencia_media:.2f}s")
        print(f"Tokens de entrada (total):      {self.input_tokens_total}")
        print(f"Tokens de saída (total):        {self.output_tokens_total}")
        print(f"Custo total estimado:           US$ {self.custo_total_usd:.6f}")
        print(f"Custo médio por pergunta:       US$ {self.custo_medio_usd:.6f}")
        print("=" * 60)