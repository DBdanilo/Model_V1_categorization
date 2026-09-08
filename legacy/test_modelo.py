# ============================================
# test_modelo.py - Teste do ambiente
# ============================================

import sys
import importlib

print("=" * 50)
print("TESTE DE AMBIENTE - MODELO DE CATEGORIZACAO")
print("=" * 50)
print(f"Python: {sys.version}")
print("=" * 50)

modulos = [
    "pandas", "numpy", "requests", "dotenv", "tqdm",
    "sentence_transformers", "torch", "faiss", "onnx",
    "onnxruntime", "transformers", "sklearn",
    "matplotlib", "seaborn"
]

falhas = 0

for nome in modulos:
    try:
        importlib.import_module(nome)
        print(f"[  OK  ] {nome}")
    except ImportError as e:
        print(f"[FALHOU] {nome}: {e}")
        falhas += 1

print("=" * 50)

if falhas == 0:
    print(">>> Todos os modulos instalados corretamente!")
else:
    print(f">>> {falhas} modulo(s) com problema.")

try:
    import numpy as np
    arr = np.array([1, 2, 3])
    print(f">>> NumPy OK - array: {arr}")
except Exception as e:
    print(f">>> NumPy falhou: {e}")
