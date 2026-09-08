# Model_V1 - Recategorização de transações com embeddings e busca vetorial

## Visão geral

Este projeto é uma prova de conceito para recategorização semântica de transações e nomes de merchants usando embeddings textuais e busca vetorial local. A ideia central é transformar descrições de transações em vetores numéricos, indexá-los em um banco vetorial com FAISS e, em seguida, encontrar o item mais semelhante para sugerir a categoria correta.

O projeto combina:

- SentenceTransformers para geração de embeddings;
- FAISS para busca por similaridade;
- um dataset próprio e um dataset externo do Hugging Face;
- scripts para geração, validação, simulação local e exportação para produção.

Em termos práticos, ele tenta responder perguntas como:

- "Este nome de estabelecimento é mais parecido com Uber, iFood ou Mercado Livre?"
- "Qual categoria esse pagamento deve receber?"
- "Qual merchant mais similar ao texto informado?"

---

## Objetivo do projeto

O objetivo principal é construir uma solução local para classificação semântica de transações financeiras, ideal para cenários em que:

- o texto da transação pode variar bastante (ex.: "Uber *UBER *TRIP", "99POP", "Uber Brasil");
- a categoria precisa ser inferida mesmo com grafias, abreviações e ruídos;
- a busca precisa ser rápida e robusta em ambiente local.

A solução atual é bem adequada para experimentação, prototipagem e geração de um índice local de busca baseado em similaridade semântica.

---

## Como o projeto funciona

A arquitetura do projeto segue este fluxo:

1. Criação de exemplos de transações e categorias.
2. Geração de embeddings com SentenceTransformer.
3. Armazenamento desses embeddings em um índice FAISS.
4. Busca por similaridade entre um texto novo e o corpus indexado.
5. Retorno da merchant mais próxima e da categoria sugerida.
6. Exportação dos artefatos para uso posterior em produção.

### Fluxo principal

- `01_criar_dataset.py` gera um dataset próprio de exemplos representativos.
- `download_hf_token.py` baixa um dataset do Hugging Face autenticado por token.
- `simular_cosmo_local.py` cria um índice sintético, gera embeddings, salva metadados e permite consultas locais.
- `models/config_producao.json` guarda a configuração do modelo em produção.
- `models/cosmo_producao.index` e `models/metadados_producao.csv` são os artefatos finais exportados para uso prático.

---

## Estrutura do repositório

```text
Model_V1/
├── 01_criar_dataset.py
├── download_hf_token.py
├── simular_cosmo_local.py
├── Requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── cosmo_service.py
│   ├── dataset_service.py
│   ├── embedding_service.py
│   └── paths.py
├── data/
│   └── raw/
│       ├── dataset_proprio.json
│       └── dataset_recategorization.json
├── models/
│   ├── config_producao.json
│   ├── cosmo_producao.index
│   ├── cosmo_simulado.index
│   ├── metadados_producao.csv
│   └── metadados_simulados.csv
├── docs/
│   └── feat.txt
├── legacy/
│   ├── deepseek_python_20260908_4b2af0.py
│   └── test_modelo.py
├── tests/
│   └── test_cosmo_service.py
├── venv/
└── __pycache__/
```

---

## Arquivos principais

### `01_criar_dataset.py`

Responsável por gerar o dataset próprio de recategorização. Ele:

- define uma lista de exemplos de merchant + categoria;
- calcula embeddings com `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")`;
- salva o resultado em `data/raw/dataset_proprio.json`.

Esse script cria um conjunto de dados pequeno, mas suficiente para testar a ideia em ambiente local.

### `download_hf_token.py`

Baixa um dataset remoto autenticado pelo Hugging Face. Ele:

- lê a variável de ambiente `HF_TOKEN`;
- faz uma requisição para a URL do dataset;
- salva em `data/raw/dataset_recategorization.json`.

Esse arquivo é útil para enriquecer ou comparar o dataset local com um dataset externo.

### `simular_cosmo_local.py`

É o script mais relevante do projeto. Ele implementa uma simulação local do que seria um sistema de busca vetorial tipo COSMO.

Ele faz:

- criação de dados sintéticos de merchants;
- geração de embeddings;
- indexação com FAISS;
- busca por similaridade de texto;
- exportação para produção.

Ele também registra as configurações do modelo em `models/config_producao.json`.

### `src/cosmo_service.py`

Centraliza a lógica reutilizável de busca vetorial:

- normalização de texto;
- criação do índice FAISS;
- busca por similaridade.

Este módulo reduz duplicação no código e facilita evolução futura.

### `src/embedding_service.py`

Responsável por carregar o modelo SentenceTransformer e gerar embeddings em um único ponto reutilizável.

### `src/dataset_service.py`

Encapsula escrita de arquivos JSON e a persistência de dados em `data/raw`.

### `src/paths.py`

Define caminhos padronizados para o projeto, evitando hardcoded paths e reduzindo erros em execuções em diferentes diretórios.

### `docs/feat.txt`

Arquivo com a definição dos campos essenciais para o pipeline de transações financeiras. Ele mapeia campos como:

- `transactionName`
- `creditDebitType`
- `amount`
- `currency`
- `transactionDateTime`
- `payeeMCC`
- `type`
- `additionalInfo`
- `partieCnpjCpf`

Esse arquivo demonstra que o projeto pretende ser integrado a um contexto de transações bancárias, com foco em extração de categoria a partir de informações do cartão e da conta.

### `legacy/`

Pasta reservada para arquivos experimentais ou rascunhos que não fazem parte do fluxo principal atual.

Hoje ela abriga itens como:

- `legacy/test_modelo.py`
- `legacy/deepseek_python_20260908_4b2af0.py`

Esses arquivos são úteis para consulta histórica, mas não são o centro da operação do projeto.

---

## Dependências

As dependências estão em `Requirements.txt`.

### Bibliotecas principais

- `pandas`
- `numpy`
- `requests`
- `python-dotenv`
- `tqdm`
- `sentence-transformers`
- `torch`
- `faiss-cpu`
- `onnx`
- `onnxruntime`
- `optimum`
- `transformers`
- `scikit-learn`
- `matplotlib`
- `seaborn`

### Modelo usado

O projeto usa:

```text
paraphrase-multilingual-MiniLM-L12-v2
```

Esse modelo é eficiente para embeddings multilíngues e funciona bem em cenários com nomes de merchant e textos em português.

---

## Instalação e ambiente

### Requisitos

- Python 3.10+ (recomendado 3.11)
- Ambiente virtual recomendado
- Acesso a internet para baixar o modelo e, caso necessário, o dataset do Hugging Face

### Passo a passo

```bash
cd c:\desenvolvimento\Model_V1
python -m venv venv
venv\Scripts\activate
pip install -r Requirements.txt
```

Se o ambiente Windows estiver usando PowerShell:

```powershell
cd c:\desenvolvimento\Model_V1
.\venv\Scripts\Activate.ps1
pip install -r Requirements.txt
```

---

## Como executar

### 1) Gerar o dataset próprio

```bash
python 01_criar_dataset.py
```

Resultado esperado:

- arquivo salvo em `data/raw/dataset_proprio.json`

### 2) Validar o modelo de embeddings

Opcionalmente, para testes históricos ou experimentais:

```bash
python legacy/test_modelo.py
```

Esse comando verifica se o modelo está carregando corretamente e se o vetor é gerado sem erro.

### 3) Baixar dataset do Hugging Face

Antes de rodar, configure a variável de ambiente:

```bash
set HF_TOKEN=seu_token_aqui
```

Depois:

```bash
python download_hf_token.py
```

### 4) Rodar a simulação local do COSMO

```bash
python simular_cosmo_local.py
```

Esse script:

- cria dados sintéticos;
- gera embeddings;
- salva o índice FAISS em `models/cosmo_simulado.index`;
- salva os metadados em `models/metadados_simulados.csv`;
- exporta uma versão de produção em `models/`.

---

## Saídas esperadas

### `data/raw/dataset_proprio.json`

Estrutura esperada:

```json
[
  {
    "description": "NETFLIX.COM",
    "category": "Streaming de vídeo/música",
    "embedding": [0.01, -0.11, ...]
  }
]
```

### `models/config_producao.json`

Exemplo de configuração exportada:

```json
{
  "versao": "1.0.0",
  "total_merchants": 1000,
  "dimensao_embedding": 384,
  "total_categorias": 11,
  "categorias": ["Entretenimento", "Viagem", "Alimentacao"],
  "limiar_confianca": 0.7,
  "modelo": "paraphrase-multilingual-MiniLM-L12-v2"
}
```

---

## Análise do projeto

### Pontos fortes

- estrutura simples e didática;
- foco claro em embeddings para recategorização;
- uso de modelos open-source e executáveis localmente;
- boa base para prototipagem sem depender de serviços externos;
- uso de FAISS torna a busca por similaridade rápida e prática.

### Limitações e pontos de melhoria

1. Falta de CLI padronizada
   - os scripts são executados diretamente em terminal, sem parâmetros, sem argparse e sem padronização de entrada/saída.

2. Risco de caminhos relativos
   - vários scripts dependem de caminhos como `models/` e `data/raw`, o que pode quebrar se forem executados de outra pasta.

3. Redundância entre scripts
   - `test_modelo.py` e `deepseek_python_20260908_4b2af0.py` parecem ser versões muito próximas e não cumprem papéis claramente distintos.

4. Falta de modularização
   - a lógica de geração de embeddings, carga do índice e busca está concentrada em scripts únicos; em uma versão mais madura, seria melhor separar em módulos e classes.

5. Ausência de testes automatizados
   - seria recomendado criar testes para validar:
     - geração do dataset;
     - dimensão dos embeddings;
     - busca do merchant mais semelhante;
     - exportação dos arquivos.

6. Falta de tratamento de erros e logging
   - há validações mínimas, mas sem estrutura robusta para cenários de token ausente, dataset inconsistente ou falha de download.

---

## Sugestões de evolução

Para transformar este protótipo em uma solução mais sólida, recomendo:

- criar uma estrutura de pacote (`src/`, `app/`, `core/`);
- centralizar a lógica de embeddings em um módulo único;
- usar `pathlib` em vez de caminhos fixos;
- adicionar `argparse` para aceitar dataset, modelo, categoria e limite de resultados;
- implementar testes com `pytest`;
- persistir model metadata de forma consistente;
- avaliar qualidade do modelo com métricas de precisão, recall e top-k;
- preparar uma API para inferência em produção.

---

## Conclusão

Este repositório funciona como um protótipo funcional de sistema local de recategorização de transações usando embeddings semânticos. O foco principal é demonstrar que nomes de merchant e descrições de pagamento podem ser comparados por similaridade semântica e convertidos em categoria sugerida com alta eficiência.

A solução já está em uma etapa valiosa de prova de conceito, mas ainda possui potencial para evoluir em direção a um sistema mais reutilizável, testável e pronto para produção.

---

## Observação final

Os arquivos de modelo gerados e os dados em `models/` e `data/raw` devem ser tratados como artefatos de experimentação. Em uma evolução do projeto, seria ideal separar claramente:

- dados brutos;
- dados processados;
- índices vetoriais;
- modelos exportados;
- artefatos de produção.

Isso torna a manutenção e a reprodução dos experimentos muito mais confiáveis.
