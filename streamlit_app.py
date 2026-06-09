from config import DATASET_PATH, BENCHMARK_FILE, OUTPUTS_DIR, LOGS_DIR
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import subprocess
import sys

from agent import Agent
from tools import state

def carregar_log_benchmark(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
        
def extrair_imagens_do_resultado(resultado):
    imagens = []

    for passo in getattr(resultado, "trajetoria", []):
        conteudo = getattr(passo, "conteudo", None)

        if isinstance(conteudo, dict):
            caminho = conteudo.get("arquivo_gerado")

            if caminho and str(caminho).endswith(".png"):
                path = Path(caminho)

                if path.exists():
                    imagens.append(path)

    return list(dict.fromkeys(imagens))

from evaluation.metrics import estimar_custo_usd


def extrair_metricas_log(dados, nome_arquivo):
    resumo = dados.get("resumo", dados.get("metricas", dados))

    resultados = (
        dados.get("resultados")
        or dados.get("perguntas")
        or dados.get("detalhes")
        or dados.get("execucoes")
        or []
    )

    total = (
        resumo.get("total_perguntas")
        or resumo.get("total")
        or len(resultados)
        or 0
    )

    acertos = (
        resumo.get("acertos")
        or resumo.get("total_acertos")
        or resumo.get("respostas_corretas")
        or sum(1 for r in resultados if r.get("correto") is True)
        or 0
    )

    acuracia = (
        resumo.get("acuracia_geral")
        or resumo.get("acuracia")
        or resumo.get("accuracy")
        or (acertos / total if total else 0)
    )

    acuracia_pct = acuracia * 100 if acuracia <= 1 else acuracia

    input_tokens = (
        resumo.get("input_tokens")
        or resumo.get("input_tokens_totais")
        or resumo.get("total_input_tokens")
        or sum(r.get("input_tokens", 0) for r in resultados)
        or 0
    )

    output_tokens = (
        resumo.get("output_tokens")
        or resumo.get("output_tokens_totais")
        or resumo.get("total_output_tokens")
        or sum(r.get("output_tokens", 0) for r in resultados)
        or 0
    )

    total_tokens = input_tokens + output_tokens

    custo_total = (
        resumo.get("custo_total")
        or resumo.get("custo_total_estimado")
        or resumo.get("custo_estimado_total")
        or resumo.get("total_cost")
        or sum(
            r.get("custo_estimado")
            or r.get("custo_total")
            or r.get("custo")
            or 0
            for r in resultados
        )
        or estimar_custo_usd(input_tokens, output_tokens)
    )

    custo_medio = (
        resumo.get("custo_medio")
        or resumo.get("custo_medio_por_pergunta")
        or resumo.get("average_cost")
        or (custo_total / total if total else 0)
    )

    tool_calls_total = (
        resumo.get("total_tool_calls")
        or sum(r.get("total_tool_calls", r.get("tool_calls", 0)) for r in resultados)
        or 0
    )

    tool_calls_media = (
        resumo.get("tool_calls_media")
        or resumo.get("media_tool_calls")
        or resumo.get("tool_calls_por_pergunta")
        or (tool_calls_total / total if total else 0)
    )

    latencia_total = (
        resumo.get("latencia_total")
        or sum(r.get("latencia_total", r.get("latencia", 0)) for r in resultados)
        or 0
    )

    latencia_media = (
        resumo.get("latencia_media")
        or resumo.get("latencia_media_s")
        or resumo.get("tempo_medio")
        or (latencia_total / total if total else 0)
    )

    return {
        "arquivo": nome_arquivo,
        "total_perguntas": total,
        "acertos": acertos,
        "acuracia_pct": acuracia_pct,
        "tool_calls_media": tool_calls_media,
        "latencia_media": latencia_media,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "custo_total": custo_total,
        "custo_medio": custo_medio,
    }

st.set_page_config(
    page_title="Agente EDA",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agente de Análise Exploratória de Dados")
st.caption("Faça perguntas em linguagem natural sobre um dataset CSV.")

# -----------------------------
# Sessão
# -----------------------------
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

if "agente" not in st.session_state:
    st.session_state.agente = None

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Configurações")

    uploaded_file = st.file_uploader(
        "Enviar dataset CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        state.df = df
        state.path = "upload_streamlit.csv"
        st.session_state.agente = Agent()
        st.success("Dataset carregado via upload.")
    else:
        if DATASET_PATH.exists():
            df = pd.read_csv(DATASET_PATH)
            state.df = df
            state.path = str(DATASET_PATH)
            st.session_state.agente = Agent()
            st.success(f"Dataset padrão carregado: {DATASET_PATH.name}")
        else:
            df = None
            st.warning("Nenhum dataset carregado.")

    if df is not None:
        st.metric("Linhas", df.shape[0])
        st.metric("Colunas", df.shape[1])

    if st.button("Limpar conversa"):
        st.session_state.mensagens = []

        for img in Path(OUTPUTS_DIR).glob("*.png"):
            try:
                img.unlink()
            except Exception:
                pass

        st.rerun()

# -----------------------------
# Abas
# -----------------------------
aba_chat, aba_dataset, aba_benchmark, aba_outputs = st.tabs(
    ["Chat", "Dataset", "Benchmark", "Outputs"]
)

# -----------------------------
# Aba Chat
# -----------------------------
with aba_chat:
    st.subheader("Conversa com o agente")

    from evaluation.metrics import estimar_custo_usd

    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg["role"] == "assistant" and "resultado" in msg:
                imagens = extrair_imagens_do_resultado(msg["resultado"])

                for img in imagens:
                    st.image(str(img), caption=img.name, use_container_width=True)
            if msg["role"] == "assistant" and "resultado" in msg:
                resultado = msg["resultado"]
                input_tokens = getattr(resultado, "input_tokens", 0)
                output_tokens = getattr(resultado, "output_tokens", 0)                       
                
                

                input_tokens = resultado.input_tokens
                output_tokens = resultado.output_tokens
                total_tokens = input_tokens + output_tokens
                custo = estimar_custo_usd(input_tokens, output_tokens)

                with st.expander("Ver trajetória das tools"):
                    for passo in resultado.trajetoria:
                        st.json(
                            {
                                "tipo": passo.tipo,
                                "conteudo": passo.conteudo,
                                "timestamp": passo.timestamp,
                            }
                        )
                col1, col2, col3 = st.columns(3)
                col1.metric("Tool calls", resultado.total_tool_calls)
                col2.metric("Iterações", resultado.total_iteracoes)
                col3.metric("Latência", f"{resultado.latencia_total:.2f}s")
                #col4.metric("Custo", f"US$ {custo:.6f}")
                # col1, col2, col3 = st.columns(3)
                # col1.metric("Tool calls", resultado.total_tool_calls)
                # col2.metric("Iterações", resultado.total_iteracoes)
                # col3.metric("Latência", f"{resultado.latencia_total:.2f}s")

                col8, col5, col6, col7 = st.columns(4)
                col8.metric("Input tokens", input_tokens)
                col5.metric("Output tokens", output_tokens)
                col6.metric("Tokens totais", total_tokens)
                col7.metric("Custo estimado", f"US$ {custo:.6f}")

    pergunta = st.chat_input("Pergunte algo sobre o dataset...")

    if pergunta:
        st.session_state.mensagens.append(
            {"role": "user", "content": pergunta}
        )

        if st.session_state.agente is None:
            resposta = "Nenhum dataset foi carregado."

            st.session_state.mensagens.append(
                {"role": "assistant", "content": resposta}
            )

            st.rerun()

        else:
            with st.spinner("Analisando..."):
                resultado = st.session_state.agente.perguntar(pergunta)

            st.session_state.mensagens.append(
                {
                    "role": "assistant",
                    "content": resultado.resposta_final,
                    "resultado": resultado,
                }
            )

            st.rerun()
# -----------------------------
# Aba Dataset
# -----------------------------
with aba_dataset:
    st.subheader("Prévia do dataset")

    if df is not None:
        st.dataframe(df.head(50), use_container_width=True)

        st.subheader("Tipos das colunas")
        tipos = pd.DataFrame(
            {
                "coluna": df.columns,
                "tipo": [str(t) for t in df.dtypes],
            }
        )
        st.dataframe(tipos, use_container_width=True)

        st.subheader("Resumo estatístico")
        st.dataframe(df.describe(include="all"), use_container_width=True)
    else:
        st.warning("Carregue um CSV para visualizar os dados.")

# -----------------------------
# Aba Benchmark
# -----------------------------
with aba_benchmark:
    st.subheader("Banco de perguntas do benchmark")

    if BENCHMARK_FILE.exists():
        with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
            dados_benchmark = json.load(f)

        if isinstance(dados_benchmark, dict):
            perguntas = dados_benchmark.get("perguntas", [])
        else:
            perguntas = dados_benchmark

        perguntas = [p for p in perguntas if isinstance(p, dict)]

        st.write(f"Total de perguntas: **{len(perguntas)}**")

        if perguntas:
            opcoes = {
                f'{p.get("id", "")} - {p.get("pergunta", "")}': p
                for p in perguntas
            }

            escolha = st.selectbox(
                "Escolha uma pergunta do benchmark",
                list(opcoes.keys()),
            )

            pergunta_escolhida = opcoes[escolha]

            st.markdown("### Pergunta")
            st.info(pergunta_escolhida.get("pergunta", ""))

            st.markdown("### Resposta esperada")

            resposta_esperada = pergunta_escolhida.get("resposta_esperada")

            if isinstance(resposta_esperada, (dict, list)):
                st.json(resposta_esperada)
            else:
                st.code(str(resposta_esperada))

            if st.button("Enviar esta pergunta para o agente"):
                pergunta_benchmark = pergunta_escolhida.get("pergunta", "")

                if st.session_state.agente is None:
                    st.error("Nenhum dataset foi carregado.")
                else:
                    with st.spinner("Executando pergunta do benchmark..."):
                        resultado = st.session_state.agente.perguntar(
                            pergunta_benchmark
                        )

                    st.markdown("### Resposta do agente")
                    st.markdown(resultado.resposta_final)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Tool calls", resultado.total_tool_calls)
                    col2.metric("Iterações", resultado.total_iteracoes)
                    col3.metric("Latência", f"{resultado.latencia_total:.2f}s")

                    with st.expander("Trajetória"):
                        for passo in resultado.trajetoria:
                            st.json(
                                {
                                    "tipo": passo.tipo,
                                    "conteudo": passo.conteudo,
                                    "timestamp": passo.timestamp,
                                }
                            )
        else:
            st.warning("Nenhuma pergunta válida encontrada no benchmark.")

        st.markdown("---")
        st.markdown("### Executar benchmark completo")

        st.warning(
            "Rodar o benchmark completo consome API/tokens. "
            "Use apenas quando quiser gerar logs finais."
        )

        if st.button("Rodar benchmark completo"):
            from pathlib import Path

            for img in Path(OUTPUTS_DIR).glob("*.png"):
                img.unlink()

            with st.spinner("Rodando benchmark..."):
                processo = subprocess.run(
                    [sys.executable, "-X", "utf8", "-m", "evaluation.benchmark"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env={
                        **__import__("os").environ,
                        "PYTHONIOENCODING": "utf-8",
                        "PYTHONUTF8": "1",
                    },
            )

            st.code(processo.stdout)

            if processo.stderr:
                if "RuntimeWarning" not in processo.stderr:
                    st.error(processo.stderr)
                    



        st.markdown("---")
        st.markdown("### Logs de benchmark")

        logs = sorted(Path(LOGS_DIR).glob("*.json"), reverse=True)

        if logs:
            st.write(f"Total de logs encontrados: **{len(logs)}**")

            nomes_logs = [log.name for log in logs]

            log_escolhido_nome = st.selectbox(
                "Escolha um JSON de benchmark para analisar",
                nomes_logs,
            )

            log_escolhido = next(
                log for log in logs if log.name == log_escolhido_nome
            )

            with open(log_escolhido, "rb") as f:
                st.download_button(
                    label=f"Baixar {log_escolhido.name}",
                    data=f,
                    file_name=log_escolhido.name,
                    mime="application/json",
                )

            try:
                dados_log = carregar_log_benchmark(log_escolhido)
                metricas = extrair_metricas_log(dados_log, log_escolhido.name)

                st.markdown("#### Estatísticas do JSON selecionado")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Perguntas", metricas["total_perguntas"])
                col2.metric("Acertos", metricas["acertos"])
                col3.metric("Acurácia", f'{metricas["acuracia_pct"]:.2f}%')
                col4.metric("Tool calls médias", f'{metricas["tool_calls_media"]:.2f}')

                col5, col6, col7, col8 = st.columns(4)
                col5.metric("Latência média", f'{metricas["latencia_media"]:.2f}s')
                col6.metric("Tokens totais", metricas["total_tokens"])
                col7.metric("Custo total", f'US$ {metricas["custo_total"]:.4f}')
                col8.metric("Custo médio", f'US$ {metricas["custo_medio"]:.4f}')

                with st.expander("Ver JSON completo"):
                    st.json(dados_log)

            except Exception as e:
                st.error(f"Erro ao analisar o log selecionado: {e}")
                
            st.markdown("#### Estatísticas de todos os Relatórios Json")

            st.markdown("#### Série temporal da acurácia")

            registros = []

            for log in sorted(logs):
                try:
                    dados = carregar_log_benchmark(log)
                    registros.append(extrair_metricas_log(dados, log.name))
                except Exception:
                    pass

            if registros:
                df_logs = pd.DataFrame(registros)

                df_logs["execucao"] = df_logs["arquivo"].str.replace(
                    "benchmark_", "", regex=False
                ).str.replace(
                    ".json", "", regex=False
                )

                st.line_chart(
                    df_logs,
                    x="execucao",
                    y="acuracia_pct",
                    use_container_width=True,
                )

                st.dataframe(
                    df_logs[
                        [
                            "arquivo",
                            "total_perguntas",
                            "acertos",
                            "acuracia_pct",
                            "tool_calls_media",
                            "latencia_media",
                            "input_tokens",
                            "output_tokens",
                            "total_tokens",
                            "custo_total",
                            "custo_medio",
                        ]
                    ],
                    use_container_width=True,
                )
            else:
                st.info("Nenhum log válido encontrado para montar a série temporal.")

        else:
            st.info("Nenhum log de benchmark encontrado.")

    else:
        st.warning("Arquivo benchmark.json não encontrado.")

# -----------------------------
# Aba Outputs
# -----------------------------
# -----------------------------
# Aba Outputs
# -----------------------------
with aba_outputs:
    st.subheader("Arquivos gerados")

    st.markdown("### Gráficos")

    import hashlib

    def hash_arquivo(path):
        return hashlib.md5(path.read_bytes()).hexdigest()

    todas_imagens = list(Path(OUTPUTS_DIR).glob("*.png"))

    vistos = set()
    imagens = []

    for img in todas_imagens:
        h = hash_arquivo(img)

        if h not in vistos:
            vistos.add(h)
            imagens.append(img)

    if imagens:
        st.info(
            f"Mostrando {len(imagens)} gráficos únicos "
            f"({len(todas_imagens)} arquivos encontrados)."
        )

        for img in imagens:
            st.image(img.open("rb").read(), caption=img.name)
    else:
        st.info("Nenhum gráfico gerado ainda.")

# ##    st.markdown("### Logs")

    # logs = list(Path(LOGS_DIR).glob("*.json"))

    # if logs:
        # for log in sorted(logs, reverse=True):
            # with open(log, "rb") as f:
                # st.download_button(
                    # label=f"Baixar {log.name}",
                    # data=f,
                    # file_name=log.name,
                    # mime="application/json",
                # )
    # else:
        # st.info("Nenhum log encontrado.")