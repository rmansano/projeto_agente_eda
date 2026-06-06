import streamlit as st
import pandas as pd
import json
from pathlib import Path
import subprocess
import sys

from agent import Agent
from tools import state
from config import DATASET_PATH, BENCHMARK_FILE, OUTPUTS_DIR, LOGS_DIR

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
                resultado = msg["resultado"]

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

                col4, col5, col6, col7 = st.columns(4)
                col4.metric("Input tokens", input_tokens)
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

    st.markdown("### Logs")

    logs = list(Path(LOGS_DIR).glob("*.json"))

    if logs:
        for log in sorted(logs, reverse=True):
            with open(log, "rb") as f:
                st.download_button(
                    label=f"Baixar {log.name}",
                    data=f,
                    file_name=log.name,
                    mime="application/json",
                )
    else:
        st.info("Nenhum log encontrado.")