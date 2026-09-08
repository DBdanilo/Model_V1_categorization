# -*- coding: utf-8 -*-
"""
01_criar_dataset.py
Gera um dataset de recategorizacao com embeddings locais (sentence-transformers)
"""

import json

from src.dataset_service import save_dataset_raw
from src.embedding_service import encode_texts, load_embedding_model

EXEMPLOS = [
    ("NETFLIX.COM", "Streaming de vídeo/música"),
    ("Spotify", "Streaming de vídeo/música"),
    ("HBO MAX", "Streaming de vídeo/música"),
    ("Disney Plus", "Streaming de vídeo/música"),
    ("Amazon Prime Video", "Streaming de vídeo/música"),
    ("Deezer", "Streaming de vídeo/música"),
    ("YouTube Premium", "Streaming de vídeo/música"),
    ("Apple Music", "Streaming de vídeo/música"),
    ("Crunchyroll", "Streaming de vídeo/música"),
    ("Uber *UBER *TRIP", "Táxi e transporte privado urbano"),
    ("99POP", "Táxi e transporte privado urbano"),
    ("Cabify", "Táxi e transporte privado urbano"),
    ("Uber Brasil", "Táxi e transporte privado urbano"),
    ("99 Motors", "Táxi e transporte privado urbano"),
    ("Táxi Rádio", "Táxi e transporte privado urbano"),
    ("iFood", "Delivery de alimentos"),
    ("Rappi Brasil", "Delivery de alimentos"),
    ("Uber Eats", "Delivery de alimentos"),
    ("Zé Delivery", "Delivery de alimentos"),
    ("Aiqfome", "Delivery de alimentos"),
    ("McDonald's", "Restaurantes, bares e lanchonetes"),
    ("BK Brasil", "Restaurantes, bares e lanchonetes"),
    ("Habib's", "Restaurantes, bares e lanchonetes"),
    ("Outback", "Restaurantes, bares e lanchonetes"),
    ("Restaurante República", "Restaurantes, bares e lanchonetes"),
    ("Giraffas", "Restaurantes, bares e lanchonetes"),
    ("Pizza Hut", "Restaurantes, bares e lanchonetes"),
    ("Burguer King", "Restaurantes, bares e lanchonetes"),
    ("Subway", "Restaurantes, bares e lanchonetes"),
    ("Posto Shell", "Postos de gasolina"),
    ("Posto Petrobras", "Postos de gasolina"),
    ("Posto Ipiranga", "Postos de gasolina"),
    ("Posto BR", "Postos de gasolina"),
    ("Posto Texaco", "Postos de gasolina"),
    ("Supermercado Pão de Açúcar", "Supermercado"),
    ("Extra Hipermercados", "Supermercado"),
    ("Carrefour", "Supermercado"),
    ("Atacadão", "Supermercado"),
    ("Mercado Municipal", "Supermercado"),
    ("Oba Hortifruti", "Supermercado"),
    ("Supermercado Guanabara", "Supermercado"),
    ("Assaí Atacadista", "Supermercado"),
    ("Drogaria São Paulo", "Farmácia e saúde"),
    ("Drogasil", "Farmácia e saúde"),
    ("Droga Raia", "Farmácia e saúde"),
    ("Pacheco", "Farmácia e saúde"),
    ("Farmácia Popular", "Farmácia e saúde"),
    ("Clínica Odontológica", "Farmácia e saúde"),
    ("Consulta médica", "Farmácia e saúde"),
    ("Alura Cursos Online", "Educação"),
    ("Coursera", "Educação"),
    ("Udemy", "Educação"),
    ("Faculdade Pitágoras", "Educação"),
    ("Curso de inglês Wizard", "Educação"),
    ("Colégio Bandeirantes", "Educação"),
    ("Hotmart", "Educação"),
    ("Eduzz", "Educação"),
    ("Aluguel apto", "Aluguel e moradia"),
    ("Condomínio", "Aluguel e moradia"),
    ("IPTU", "Aluguel e moradia"),
    ("Imobiliária", "Aluguel e moradia"),
    ("Prestação da casa", "Aluguel e moradia"),
    ("Energia elétrica", "Contas de consumo"),
    ("Conta de luz", "Contas de consumo"),
    ("Água e esgoto", "Contas de consumo"),
    ("Internet fibra", "Contas de consumo"),
    ("NET Claro", "Contas de consumo"),
    ("Vivo telefonia", "Contas de consumo"),
    ("Tim celular", "Contas de consumo"),
    ("Gás encanado", "Contas de consumo"),
    ("Cinema Kinoplex", "Lazer e entretenimento"),
    ("Steam Games", "Lazer e entretenimento"),
    ("Xbox Live", "Lazer e entretenimento"),
    ("PlayStation Store", "Lazer e entretenimento"),
    ("Livraria Cultura", "Lazer e entretenimento"),
    ("Amazon Livros", "Lazer e entretenimento"),
    ("Show musical", "Lazer e entretenimento"),
    ("Teatro Santander", "Lazer e entretenimento"),
    ("Salário", "Receitas"),
    ("Pix recebido", "Receitas"),
    ("Transferência recebida", "Receitas"),
    ("Restituição IR", "Receitas"),
    ("Vale alimentação", "Receitas"),
    ("Rendimento investimento", "Receitas"),
    ("Freela projeto", "Receitas"),
    ("Reembolso empresa", "Receitas"),
    ("Mercado Livre", "Compras online"),
    ("Amazon Marketplace", "Compras online"),
    ("Shopee", "Compras online"),
    ("Magazine Luiza", "Compras online"),
    ("Casas Bahia", "Compras online"),
    ("Americanas.com", "Compras online"),
    ("Roupas Renner", "Compras"),
    ("C&A", "Compras"),
    ("Zara Brasil", "Compras"),
    ("Farm delivery", "Compras"),
    ("Papelaria", "Compras"),
]

def main():
    print("=" * 60)
    print("GERANDO DATASET PROPRIO DE RECATEGORIZACAO")
    print("=" * 60)

    transacoes = []
    for desc, cat in EXEMPLOS:
        transacoes.append({
            "description": desc,
            "category": cat,
            "embedding": None
        })

    print(f"Exemplos base: {len(transacoes)}")

    categorias = {}
    for item in transacoes:
        cat = item["category"]
        categorias[cat] = categorias.get(cat, 0) + 1

    print(f"Categorias ({len(categorias)}):")
    for cat, count in sorted(categorias.items(), key=lambda x: -x[1]):
        print(f"   - {cat}: {count}")

    print("\nCarregando modelo de embeddings...")
    modelo = load_embedding_model()

    print("Calculando embeddings...")
    descricoes = [item["description"] for item in transacoes]
    embeddings = encode_texts(modelo, descricoes, normalize_embeddings=True, show_progress_bar=True)

    for item, emb in zip(transacoes, embeddings):
        item["embedding"] = emb.tolist()

    print(f"Embeddings calculados: {len(embeddings)} vetores")
    print(f"Dimensao: {len(transacoes[0]['embedding'])}")

    output_file = save_dataset_raw(transacoes, "dataset_proprio.json")

    print(f"Dataset salvo em: {output_file}")

if __name__ == "__main__":
    main()
