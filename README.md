# Agente de Análise Exploratória de Dados (EDA) com LLM e Tool Use

Trabalho Final — Processamento de Linguagem Natural (PLN)
FATEC Ourinhos — Tecnologia em Ciência de Dados

---

## Sobre o Projeto

Este projeto implementa um agente conversacional capaz de responder perguntas em linguagem natural sobre um dataset CSV.

O agente utiliza um modelo de linguagem (LLM) aliado a ferramentas especializadas (*tools*) para realizar análises exploratórias de dados (EDA), executar cálculos, gerar gráficos e produzir insights automaticamente.

Fluxo de funcionamento:

Usuário → Agente → LLM → Tool → Observação → Resposta Final

O modelo decide autonomamente quais ferramentas utilizar para responder cada pergunta.

---

## Funcionalidades

### Inspeção de Dados

* Listar colunas do dataset
* Descrever dados
* Contar valores
* Contar valores nulos

### Filtragem e Agregação

* Filtrar registros
* Agrupar e agregar dados

### Estatística

* Correlação entre variáveis
* Detecção de outliers (IQR e Z-Score)
* Percentis

### Visualização

* Histograma
* Boxplot
* Scatter Plot
* Bar Plot
* Gráfico de Linha

### Ferramentas Extras

* Diagnóstico automático do dataset
* Detecção automática de correlações
* Geração automática de insights
* Tabela de contingência

---

## Estrutura do Projeto

```text
projeto_agente_eda/
│
├── agent/
├── tools/
├── evaluation/
├── tests/
├── data/
├── logs/
├── outputs/
│
├── cli.py
├── config.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

---

## Dataset Utilizado

Dataset de vendas de e-commerce contendo:

* 1.500 registros
* 10 colunas

Principais atributos:

* pedido_id
* idade_cliente
* cidade
* categoria
* tipo_cliente
* forma_pagamento
* quantidade
* preco_unitario
* desconto
* valor_total

---

## Benchmark

O projeto possui um benchmark próprio para avaliação do agente.

Total de perguntas:

* 10 factuais
* 15 analíticas
* 5 ambíguas/inválidas

Métricas calculadas:

* Acurácia
* Taxa de execução bem-sucedida
* Latência média
* Tool calls médias
* Tokens de entrada
* Tokens de saída
* Custo estimado

Resultados obtidos:

* Até 100% de acerto
* 100% de taxa de execução bem-sucedida

---

## Instalação

### 1. Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd projeto_agente_eda
```

### 2. Criar ambiente virtual

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/Mac:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar API

Copie o arquivo:

```bash
copy .env.example .env
```

Edite o arquivo `.env`:

```env
DEEPSEEK_API_KEY=SUA_CHAVE_AQUI
DEEPSEEK_BASE_URL=https://api.deepseek.com
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
```

---

## Execução

### CLI

```bash
python cli.py
```

### Interface Streamlit

```bash
streamlit run streamlit_app.py
```


### Benchmark

```bash
python -m evaluation.benchmark
```

### Testes

```bash
pytest tests/
```

---

## Interface Streamlit

A interface web permite:

* Upload de datasets CSV
* Chat conversacional com o agente
* Visualização do dataset
* Consulta ao banco de perguntas do benchmark
* Execução do benchmark completo
* Visualização de gráficos gerados
* Download de logs

---

## Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Streamlit
* DeepSeek API
* Rich
* Pytest

---

## Observações

* O arquivo `.env` não deve ser compartilhado.
* O projeto utiliza Tool Use para acesso controlado ao dataset.
* Logs e gráficos gerados são armazenados nas pastas `logs/` e `outputs/`.
* O benchmark foi desenvolvido para avaliar respostas factuais, analíticas e ambíguas.

---

## Autor

Projeto desenvolvido como Trabalho Final da disciplina de Processamento de Linguagem Natural (PLN) — FATEC Ourinhos.
