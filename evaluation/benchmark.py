"""
Executor do benchmark.

Lê o arquivo benchmark.json, roda cada pergunta no agente, compara com o
gabarito e produz um relatório agregado em logs/.

Executar com:
    python -m evaluation.benchmark
"""

import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from agent import Agent
from tools import state
from config import BENCHMARK_FILE, DATASET_PATH, LOGS_DIR
from .metrics import avaliar_resposta, BenchmarkSummary, estimar_custo_usd


console = Console()


def carregar_benchmark(caminho: Path = BENCHMARK_FILE) -> list[dict]:
    """Lê o JSON do benchmark e retorna a lista de perguntas."""
    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)
    return dados.get("perguntas", [])


def rodar_benchmark():
    # ============ 1. Carregar dataset ============
    if not Path(DATASET_PATH).exists():
        console.print(f"[red]Dataset não encontrado: {DATASET_PATH}[/red]")
        console.print("Ajuste DATASET_PATH em config.py.")
        return

    state.load(str(DATASET_PATH))
    console.print(
        f"[green]✓[/green] Dataset: {DATASET_PATH.name} "
        f"({len(state.df)} linhas × {len(state.df.columns)} colunas)"
    )

    # ============ 2. Carregar perguntas ============
    perguntas = carregar_benchmark()
    if not perguntas:
        console.print("[red]Benchmark vazio![/red]")
        return

    console.print(f"[green]✓[/green] Benchmark carregado: {len(perguntas)} perguntas\n")

    # ============ 3. Inicializar agente ============
    try:
        agente = Agent()
    except RuntimeError as e:
        console.print(f"[red]Erro:[/red] {e}")
        return

    # ============ 4. Rodar cada pergunta ============
    resultados_brutos: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Avaliando", total=len(perguntas))

        for p in perguntas:
            progress.update(task, description=f"[{p['id']}] {p['pergunta'][:50]}...")

            resultado = agente.perguntar(p["pergunta"])

            # Avalia se a resposta está correta
            correto = False
            if resultado.sucesso and p.get("resposta_esperada") is not None:
                try:
                    correto = avaliar_resposta(
                        resultado.resposta_final,
                        p["resposta_esperada"],
                        p["tipo_resposta"],
                    )
                except Exception as e:
                    console.print(f"[yellow]Falha na avaliação de {p['id']}: {e}[/yellow]")

            custo_usd = estimar_custo_usd(resultado.input_tokens, resultado.output_tokens)

            resultados_brutos.append({
                "id": p["id"],
                "tipo": p["tipo"],
                "pergunta": p["pergunta"],
                "resposta_esperada": p.get("resposta_esperada"),
                "resposta_obtida": resultado.resposta_final,
                "correto": correto,
                "execucao_sucesso": resultado.sucesso,
                "tool_calls": resultado.total_tool_calls,
                "iteracoes": resultado.total_iteracoes,
                "input_tokens": resultado.input_tokens,
                "output_tokens": resultado.output_tokens,
                "custo_usd": custo_usd,
                "latencia_seg": resultado.latencia_total,
                "trajetoria": [
                    {"tipo": s.tipo, "conteudo": s.conteudo}
                    for s in resultado.trajetoria
                ],
            })

            progress.advance(task)

    # ============ 5. Agregar métricas ============
    total = len(resultados_brutos)
    acertos = sum(1 for r in resultados_brutos if r["correto"])
    execucoes_ok = sum(1 for r in resultados_brutos if r["execucao_sucesso"])

    # Acurácia por tipo
    por_tipo_correto = defaultdict(int)
    por_tipo_total = defaultdict(int)
    for r in resultados_brutos:
        por_tipo_total[r["tipo"]] += 1
        if r["correto"]:
            por_tipo_correto[r["tipo"]] += 1

    acc_por_tipo = {
        tipo: por_tipo_correto[tipo] / por_tipo_total[tipo]
        for tipo in por_tipo_total
    }

    resumo = BenchmarkSummary(
        total_perguntas=total,
        acertos=acertos,
        taxa_execucao_sucesso=execucoes_ok / total,
        acuracia_geral=acertos / total,
        acuracia_por_tipo=acc_por_tipo,
        tool_calls_media=sum(r["tool_calls"] for r in resultados_brutos) / total,
        latencia_media=sum(r["latencia_seg"] for r in resultados_brutos) / total,
        input_tokens_total=sum(r["input_tokens"] for r in resultados_brutos),
        output_tokens_total=sum(r["output_tokens"] for r in resultados_brutos),
        custo_total_usd=sum(r["custo_usd"] for r in resultados_brutos),
        custo_medio_usd=sum(r["custo_usd"] for r in resultados_brutos) / total,
    )

    resumo.imprimir()

    # ============ 6. Salvar log completo ============
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"benchmark_{ts}.json"
    log_data = {
        "timestamp": ts,
        "dataset": str(DATASET_PATH),
        "resumo": {
            "total_perguntas": resumo.total_perguntas,
            "acertos": resumo.acertos,
            "acuracia_geral": resumo.acuracia_geral,
            "taxa_execucao_sucesso": resumo.taxa_execucao_sucesso,
            "acuracia_por_tipo": resumo.acuracia_por_tipo,
            "tool_calls_media": resumo.tool_calls_media,
            "latencia_media": resumo.latencia_media,
            "input_tokens_total": resumo.input_tokens_total,
            "output_tokens_total": resumo.output_tokens_total,
            "custo_total_usd": resumo.custo_total_usd,
            "custo_medio_usd": resumo.custo_medio_usd,
        },
        "resultados": resultados_brutos,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2, default=str)

    console.print(f"\n[green]✓[/green] Log completo salvo em [bold]{log_path}[/bold]")


if __name__ == "__main__":
    rodar_benchmark()
