# scripts/simular_cosmo_local.py
"""
Simula um COSMO (FAISS) local com busca vetorial
"""

import json
import os
import random
import time

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.cosmo_service import build_index, normalize_text, search_similar
from src.paths import MODELS_DIR

# ============================================
# PASSO 1: CRIAR DADOS SINTÉTICOS (OU CARREGAR EXISTENTES)
# ============================================

def criar_dados_sinteticos():
    """
    Cria dados sintéticos de transações para simulação
    """
    print("📊 Criando dados sintéticos...")
    
    # Lista de merchants reais (exemplos)
    merchants_reais = [
        "UBER DO BRASIL", "IFOOD COM BR", "MERCADO PAGO",
        "AMAZON BRASIL", "PAGBANCO", "NETSHOES",
        "PADARIA DO SEU ZE", "POSTO IPIRANGA", "MAGAZINE LUIZA",
        "CARREFOUR", "AMERICANAS", "RIACHUELO",
        "MCDONALDS", "BURGER KING", "SUBWAY",
        "FARMACIA SAO PAULO", "DROGARIA RAIA", "DROGASIL",
        "CINEMARK", "PLAYARTE", "CINESYSTEM",
        "HOTEL TRANSMERICA", "HOTEL TIVOLI", "AIRBNB",
        "99 TAXI", "LOGGI", "CABIFY",
        "SPOTIFY", "NETFLIX", "AMAZON PRIME",
        "GLOBO PLAY", "HBO MAX", "DISNEY PLUS",
        "SAMSUNG BRASIL", "APPLE BRASIL", "MICROSOFT",
        "GITHUB", "AWS BRASIL", "GOOGLE CLOUD"
    ]
    
    # Categorias reais
    categorias = {
        "UBER DO BRASIL": "Transporte",
        "IFOOD COM BR": "Alimentacao",
        "MERCADO PAGO": "Fintech",
        "AMAZON BRASIL": "Varejo",
        "PAGBANCO": "Fintech",
        "NETSHOES": "Varejo",
        "PADARIA DO SEU ZE": "Alimentacao",
        "POSTO IPIRANGA": "Combustivel",
        "MAGAZINE LUIZA": "Varejo",
        "CARREFOUR": "Supermercado",
        "AMERICANAS": "Varejo",
        "RIACHUELO": "Varejo",
        "MCDONALDS": "Alimentacao",
        "BURGER KING": "Alimentacao",
        "SUBWAY": "Alimentacao",
        "FARMACIA SAO PAULO": "Saude",
        "DROGARIA RAIA": "Saude",
        "DROGASIL": "Saude",
        "CINEMARK": "Lazer",
        "PLAYARTE": "Lazer",
        "CINESYSTEM": "Lazer",
        "HOTEL TRANSMERICA": "Viagem",
        "HOTEL TIVOLI": "Viagem",
        "AIRBNB": "Viagem",
        "99 TAXI": "Transporte",
        "LOGGI": "Transporte",
        "CABIFY": "Transporte",
        "SPOTIFY": "Entretenimento",
        "NETFLIX": "Entretenimento",
        "AMAZON PRIME": "Entretenimento",
        "GLOBO PLAY": "Entretenimento",
        "HBO MAX": "Entretenimento",
        "DISNEY PLUS": "Entretenimento",
        "SAMSUNG BRASIL": "Varejo",
        "APPLE BRASIL": "Varejo",
        "MICROSOFT": "Servicos",
        "GITHUB": "Servicos",
        "AWS BRASIL": "Servicos",
        "GOOGLE CLOUD": "Servicos"
    }
    
    # Gera 1000 transações com variações
    dados = []
    for i in range(1000):
        merchant_base = random.choice(merchants_reais)
        
        # Adiciona variações nos nomes
        sufixos = ["", " SA", " LTDA", " COM", " *TRIP", " PAG", " RECEB"]
        merchant = merchant_base + random.choice(sufixos)
        
        # Adiciona algumas variações com números
        if random.random() > 0.7:
            merchant += f" {random.randint(1, 999)}"
        
        # Pega a categoria correta
        categoria_base = categorias.get(merchant_base, "Outros")
        
        # Adiciona um pequeno ruído nas categorias (5%)
        if random.random() < 0.05:
            outras_categorias = list(set(categorias.values()))
            outras_categorias.remove(categoria_base)
            categoria = random.choice(outras_categorias)
        else:
            categoria = categoria_base
        
        dados.append({
            "merchant_name": merchant,
            "categoria": categoria
        })
    
    df = pd.DataFrame(dados)
    print(f"✅ {len(df)} transações sintéticas criadas")
    return df

# ============================================
# PASSO 2: GERAR EMBEDDINGS E CRIAR FAISS
# ============================================

def criar_cosmo_local(df):
    """
    Cria o COSMO (FAISS) local a partir dos dados
    """
    print("\n🔢 Criando COSMO local...")
    
    # 1. Carrega o modelo
    print("🔄 Carregando modelo de embedding...")
    modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 2. Gera embeddings
    textos = df['merchant_name'].tolist()
    print(f"🔄 Gerando embeddings para {len(textos)} merchants...")
    
    inicio = time.time()
    embeddings = modelo.encode(textos, show_progress_bar=True)
    tempo = time.time() - inicio
    
    embeddings = np.array(embeddings).astype('float32')
    print(f"✅ Embeddings gerados em {tempo:.2f} segundos")
    print(f"📐 Dimensão: {embeddings.shape}")
    
    # 3. Cria índice FAISS
    print("🔄 Criando índice FAISS...")
    dimensao = embeddings.shape[1]
    indice, indexed_records = build_index(df.to_dict("records"), embeddings)
    
    print(f"✅ {indice.ntotal} vetores indexados")
    
    # 4. Salva o índice
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(indice, str(MODELS_DIR / "cosmo_simulado.index"))
    df.to_csv(MODELS_DIR / "metadados_simulados.csv", index=False)
    
    print(f"💾 COSMO salvo em: {MODELS_DIR / 'cosmo_simulado.index'}")
    print(f"💾 Metadados salvos em: {MODELS_DIR / 'metadados_simulados.csv'}")
    
    return indice, df

# ============================================
# PASSO 3: FUNÇÃO DE BUSCA
# ============================================

def buscar_no_cosmo(texto_busca, indice=None, df=None, k=1):
    """
    Busca no COSMO e retorna o merchant mais parecido
    """
    # Carrega se não foi passado
    if indice is None:
        indice = faiss.read_index(str(MODELS_DIR / "cosmo_simulado.index"))
    
    if df is None:
        df = pd.read_csv(MODELS_DIR / "metadados_simulados.csv")
    
    # 1. Gera embedding do texto de busca
    modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    texto_limpo = normalize_text(texto_busca)
    vetor = modelo.encode([texto_limpo]).astype('float32')
    
    # 2. Busca no FAISS usando a camada reutilizável
    resultados_brutos = search_similar(vetor[0], indice, df.to_dict("records"), k=k)

    resultados = []
    for item in resultados_brutos:
        resultados.append({
            "merchant": item["merchant_name"],
            "categoria": item["categoria"],
            "distancia": float(item["distancia"]),
            "confianca": 1.0 / (1.0 + item["distancia"])
        })
    
    return resultados

# ============================================
# PASSO 4: TESTAR A SIMULAÇÃO
# ============================================

def testar_simulacao():
    """
    Testa a simulação do COSMO local
    """
    print("="*70)
    print("🧪 TESTANDO COSMO LOCAL")
    print("="*70)
    
    # 1. Carrega ou cria dados
    if os.path.exists(MODELS_DIR / "metadados_simulados.csv"):
        print("📂 Carregando dados existentes...")
        df = pd.read_csv(MODELS_DIR / "metadados_simulados.csv")
        indice = faiss.read_index(str(MODELS_DIR / "cosmo_simulado.index"))
        print(f"✅ {len(df)} merchants carregados")
    else:
        print("📊 Criando novos dados...")
        df = criar_dados_sinteticos()
        indice, df = criar_cosmo_local(df)
    
    # 2. Testes de busca
    print("\n" + "="*70)
    print("🔍 TESTES DE BUSCA")
    print("="*70)
    
    testes = [
        "DROGASIL",
    ]
    
    for texto in testes:
        print(f"\n🔎 Buscando: '{texto}'")
        resultados = buscar_no_cosmo(texto, indice, df, k=3)
        
        for i, resultado in enumerate(resultados):
            confianca = resultado['confianca'] * 100
            print(f"   {i+1}º: {resultado['merchant']}")
            print(f"      Categoria: {resultado['categoria']}")
            print(f"      Distância: {resultado['distancia']:.4f}")
            print(f"      Confiança: {confianca:.1f}%")
        
        # Sugestão da melhor categoria
        melhor = resultados[0]
        print(f"   ✅ Sugestão: {melhor['categoria']} (confiança: {melhor['confianca']*100:.1f}%)")
    
    # 3. Estatísticas gerais
    print("\n" + "="*70)
    print("📊 ESTATÍSTICAS DO COSMO")
    print("="*70)
    print(f"Total de merchants: {len(df)}")
    print(f"Dimensão dos embeddings: {indice.d}")
    print(f"Categorias: {df['categoria'].nunique()}")
    print("\nDistribuição por categoria:")
    print(df['categoria'].value_counts())
    
    return indice, df

# ============================================
# PASSO 5: EXPORTAR PARA PRODUÇÃO (OPCIONAL)
# ============================================

def exportar_para_producao(indice, df):
    """
    Exporta os arquivos para produção
    """
    print("\n" + "="*70)
    print("📦 EXPORTANDO PARA PRODUÇÃO")
    print("="*70)
    
    # 1. Salva o índice
    faiss.write_index(indice, str(MODELS_DIR / "cosmo_producao.index"))
    
    # 2. Salva metadados
    df.to_csv(MODELS_DIR / "metadados_producao.csv", index=False)
    
    # 3. Salva configuração
    config = {
        "versao": "1.0.0",
        "total_merchants": len(df),
        "dimensao_embedding": indice.d,
        "total_categorias": df['categoria'].nunique(),
        "categorias": df['categoria'].unique().tolist(),
        "limiar_confianca": 0.7,
        "modelo": "paraphrase-multilingual-MiniLM-L12-v2"
    }
    
    with open(MODELS_DIR / "config_producao.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("✅ Arquivos exportados:")
    print(f"   - {MODELS_DIR / 'cosmo_producao.index'}")
    print(f"   - {MODELS_DIR / 'metadados_producao.csv'}")
    print(f"   - {MODELS_DIR / 'config_producao.json'}")

# ============================================
# EXECUTAR TUDO
# ============================================

if __name__ == "__main__":
    # 1. Testa a simulação
    indice, df = testar_simulacao()
    
    # 2. Exporta para produção
    exportar_para_producao(indice, df)
    
    print("\n" + "="*70)
    print("✅ SIMULAÇÃO DO COSMO LOCAL CONCLUÍDA!")
    print("="*70)