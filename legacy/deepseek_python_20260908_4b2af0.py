# test_modelo.py
from sentence_transformers import SentenceTransformer
import numpy as np

print("🔄 Carregando modelo...")
modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("🔄 Testando com texto...")
texto = "MERCADO PAGO SA"
vetor = modelo.encode(texto)

print(f"✅ Texto: {texto}")
print(f"✅ Dimensão do vetor: {len(vetor)}")
print(f"✅ Primeiros 5 números: {vetor[:5]}")

print("\n✅ Modelo funcionando perfeitamente!")