import os

import requests

from src.dataset_service import save_dataset_raw
from src.paths import DATA_RAW_DIR

HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    raise SystemExit("❌ Token não encontrado. Defina a variável HF_TOKEN antes de rodar.")

url = "https://huggingface.co/datasets/leticiamantovani/recategorization/resolve/main/embeddings.json"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

print("Baixando dataset...")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print(f"✅ Dataset baixado! {len(data)} exemplos")

    output_file = save_dataset_raw(data, "dataset_recategorization.json")
    print(f"💾 Salvo em {output_file}")

    # Exploração básica
    primeiro = data[0]
    print("\n📋 Primeiro exemplo:")
    print(f"   Descrição: {primeiro.get('description')}")
    print(f"   Categoria: {primeiro.get('category')}")
    print(f"   Embedding: {len(primeiro.get('embedding', []))} dimensões")

    categorias = {}
    for item in data:
        cat = item.get("category", "desconhecida")
        categorias[cat] = categorias.get(cat, 0) + 1

    print(f"\n📂 Categorias ({len(categorias)}):")
    for cat, count in sorted(categorias.items(), key=lambda x: -x[1]):
        print(f"   - {cat}: {count}")

else:
    print(f"❌ Erro {response.status_code}:")
    print(response.text[:500])
