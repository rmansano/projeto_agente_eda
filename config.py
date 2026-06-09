"""
Configurações centralizadas do projeto.

Tudo que pode variar entre execuções (caminhos, modelo, limites) fica aqui.
Assim os alunos não precisam caçar valores espalhados pelo código.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis do .env automaticamente
load_dotenv()

# ============================================================
# CAMINHOS
# ============================================================
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = ROOT_DIR / "logs"
EVAL_DIR = ROOT_DIR / "evaluation"

# Garante que as pastas existem
for d in [DATA_DIR, OUTPUTS_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)

# ============================================================
# DATASET
# ============================================================
# Dataset escolhido pelo grupo: vendas/e-commerce (amostra com 1.500 registros).
DATASET_PATH = DATA_DIR / "vendas_ecommerce.csv"

# ============================================================
# LLM
# ============================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")

# Chaves de API (lidas do .env)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ============================================================
# AGENTE
# ============================================================
# Quantas iterações no máximo o agente pode fazer antes de desistir.
# Evita loops infinitos quando o agente fica confuso.
MAX_AGENT_ITERATIONS = 10

# Quantos tokens no máximo o LLM pode gerar por resposta.
MAX_TOKENS_PER_RESPONSE = 1024

# ============================================================
# AVALIAÇÃO
# ============================================================
BENCHMARK_FILE = EVAL_DIR / "benchmark.json"

# Tolerância numérica para considerar duas respostas iguais
# (ex.: 100.0 e 100.001 devem ser tratadas como iguais)
NUMERIC_TOLERANCE = 1e-2


# ============================================================
# CUSTO ESTIMADO DA API
# ============================================================
# Valores aproximados, ajustáveis conforme o modelo usado pelo grupo.
# Estão em US$ por 1 milhão de tokens.
INPUT_COST_PER_1M_TOKENS = float(os.getenv("INPUT_COST_PER_1M_TOKENS", "0.14"))
OUTPUT_COST_PER_1M_TOKENS = float(os.getenv("OUTPUT_COST_PER_1M_TOKENS", "0.28"))
