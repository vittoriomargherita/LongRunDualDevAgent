# Setup Guide - Quick Start

## 🚀 Setup Rapido

### 1. Prerequisiti

Assicurati di avere installato:
- Python 3.10+
- pip
- Git

### 2. Installazione

```bash
# 1. Crea ambiente virtuale
python3 -m venv venv

# 2. Attiva ambiente virtuale
source venv/bin/activate  # Linux/macOS
# oppure
venv\Scripts\activate  # Windows

# 3. Installa dipendenze
pip install -r requirements.txt
```

### 3. Configurazione

```bash
# Copia il file di esempio
cp config.json.example config.json

# Modifica config.json con le tue impostazioni
nano config.json  # o il tuo editor preferito
```

### 4. Configura Server LLM

Assicurati che i server LLM siano in esecuzione:

```bash
# Server Planner (porta 8081)
python -m llama_cpp.server --model path/to/planner-model.gguf --port 8081

# Server Executor (porta 8080)
python -m llama_cpp.server --model path/to/executor-model.gguf --port 8080
```

### 5. Prepara il Task

Modifica `input/task.txt` con la descrizione del software da sviluppare.

### 6. Avvia l'Agente

```bash
# Metodo 1: Script di avvio
./run_agent.sh

# Metodo 2: Manuale
source venv/bin/activate
python3 code_agent.py
```

## 📝 Prima di Committare su Git

### Checklist Pre-Commit

- [ ] `config.json` contiene solo valori fittizi (non token reali)
- [ ] `config.json` è nel `.gitignore` (già configurato)
- [ ] `output/` è nel `.gitignore` (già configurato)
- [ ] `venv/` è nel `.gitignore` (già configurato)
- [ ] File sensibili sono esclusi

### Verifica File da Committare

```bash
# Verifica cosa verrebbe committato
git status

# Dovresti vedere:
# - code_agent.py
# - requirements.txt
# - README.md
# - config.json.example
# - .gitignore
# - .gitattributes
# - input/task.txt (opzionale)
# - run_agent.sh

# NON dovresti vedere:
# - config.json (contenente token reali)
# - output/
# - venv/
```

### Inizializza Repository Git

```bash
# Inizializza repository (solo se non esiste)
git init

# Aggiungi file
git add .

# Verifica cosa viene aggiunto
git status

# Commit iniziale
git commit -m "Initial commit: AI Development Agent"

# Aggiungi remote (opzionale)
git remote add origin <your-repo-url>

# Push (opzionale)
git push -u origin main
```

## ⚠️ Importante

**NON committare mai:**
- `config.json` con token reali
- `output/` (ha il suo repository Git)
- `venv/` (ambiente virtuale)
- File con dati sensibili

Tutti questi file sono già nel `.gitignore`.



