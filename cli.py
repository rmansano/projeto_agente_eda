"""
Interface de linha de comando (CLI) do agente.

Execute com:
    python cli.py

Comandos especiais durante a sessão:
    /sair         - encerra
    /trajetoria   - mostra a trajetória da última pergunta
    /custo        - mostra custo/tokens acumulados na sessão
    /ajuda        - lista de comandos
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent import Agent
from tools import state
from config import DATASET_PATH


console = Console()


def imprimir_boas_vindas():
    console.print(Panel.fit(
        "[bold blue]Agente EDA[/bold blue] — Análise de CSV em linguagem natural\n"
        "Digite sua pergunta e pressione Enter. /ajuda para comandos.",
        border_style="blue",
    ))


def imprimir_ajuda():
    tabela = Table(title="Comandos disponíveis")
    tabela.add_column("Comando", style="cyan")
    tabela.add_column("Descrição")
    tabela.add_row("/sair", "Encerra a sessão")
    tabela.add_row("/trajetoria", "Mostra a trajetória da última pergunta")
    tabela.add_row("/custo", "Mostra tokens e tempo acumulados")
    tabela.add_row("/ajuda", "Esta tabela")
    console.print(tabela)


def imprimir_trajetoria(resultado):
    if resultado is None:
        console.print("[yellow]Sem trajetória — faça uma pergunta primeiro.[/yellow]")
        return

    tabela = Table(title=f"Trajetória: {resultado.pergunta[:60]}")
    tabela.add_column("#", style="dim", width=3)
    tabela.add_column("Tipo", style="cyan")
    tabela.add_column("Conteúdo")

    for i, passo in enumerate(resultado.trajetoria, start=1):
        if passo.tipo == "llm_text":
            conteudo = passo.conteudo[:120] + ("..." if len(passo.conteudo) > 120 else "")
        elif passo.tipo == "tool_call":
            args_str = ", ".join(f"{k}={v}" for k, v in passo.conteudo["argumentos"].items())
            conteudo = f"[green]{passo.conteudo['nome']}[/green]({args_str})"
        else:  # tool_result
            conteudo = str(passo.conteudo)[:120]
        tabela.add_row(str(i), passo.tipo, conteudo)

    console.print(tabela)


def main():
    # 1. Verifica que o dataset existe
    if not Path(DATASET_PATH).exists():
        console.print(
            f"[red]Erro:[/red] dataset não encontrado em {DATASET_PATH}.\n"
            "Coloque seu CSV na pasta data/ e ajuste DATASET_PATH em config.py."
        )
        sys.exit(1)

    # 2. Carrega o dataset
    state.load(str(DATASET_PATH))
    console.print(
        f"[green]✓[/green] Dataset carregado: [bold]{DATASET_PATH.name}[/bold] "
        f"({len(state.df)} linhas × {len(state.df.columns)} colunas)\n"
    )

    # 3. Inicializa o agente
    try:
        agente = Agent()
    except RuntimeError as e:
        console.print(f"[red]Erro:[/red] {e}")
        sys.exit(1)

    imprimir_boas_vindas()

    # 4. Sessão interativa
    ultima_resposta = None
    custo_acumulado = {"input": 0, "output": 0, "latencia": 0.0, "tool_calls": 0}

    while True:
        try:
            pergunta = console.input("\n[bold cyan]> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Até mais.[/dim]")
            break

        if not pergunta:
            continue

        # Comandos especiais
        if pergunta == "/sair":
            console.print("[dim]Até mais.[/dim]")
            break
        elif pergunta == "/ajuda":
            imprimir_ajuda()
            continue
        elif pergunta == "/trajetoria":
            imprimir_trajetoria(ultima_resposta)
            continue
        elif pergunta == "/custo":
            console.print(
                f"Tokens entrada: {custo_acumulado['input']}\n"
                f"Tokens saída:   {custo_acumulado['output']}\n"
                f"Tool calls:     {custo_acumulado['tool_calls']}\n"
                f"Latência total: {custo_acumulado['latencia']:.2f}s"
            )
            continue

        # Pergunta normal — chama o agente
        with console.status("[dim]Pensando...[/dim]"):
            resultado = agente.perguntar(pergunta)

        ultima_resposta = resultado
        custo_acumulado["input"] += resultado.input_tokens
        custo_acumulado["output"] += resultado.output_tokens
        custo_acumulado["latencia"] += resultado.latencia_total
        custo_acumulado["tool_calls"] += resultado.total_tool_calls

        # Imprime resposta
        cor_borda = "green" if resultado.sucesso else "red"
        console.print(Panel(
            resultado.resposta_final,
            border_style=cor_borda,
            title=f"[dim]{resultado.total_tool_calls} tool calls · "
                  f"{resultado.latencia_total:.2f}s · "
                  f"{resultado.input_tokens + resultado.output_tokens} tokens[/dim]",
        ))


if __name__ == "__main__":
    main()
