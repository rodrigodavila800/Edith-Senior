from tokenizers import ByteLevelBPETokenizer
import os

# Vamos varrer TODAS as pastas que você baixou em data/
base_path = "data"
paths = []

print("🔍 Mapeando a galáxia Laravel...")
for root, dirs, files in os.walk(base_path):
    for file in files:
        # Pega PHP (Código), MD (Documentação) e JSON (Configurações)
        if file.endswith((".php", ".md", ".json")):
            paths.append(os.path.join(root, file))

print(f"📚 Encontrados {len(paths)} arquivos para a Edith estudar!")

# Inicializa o tradutor (Dicionário de 52 mil "palavras" de código)
tokenizer = ByteLevelBPETokenizer()

# Treina o Tokenizer (Isso vai usar o poder do seu i7)
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>", "<pad>", "</s>", "<unk>", "<mask>",
])

# Salva o resultado
os.makedirs("edith-laravel-grand", exist_ok=True)
tokenizer.save_model("edith-laravel-grand")
print("✅ SUCESSO! O alfabeto da Arquiteta Laravel foi criado.")
