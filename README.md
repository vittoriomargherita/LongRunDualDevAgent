# AI Development Agent - Autonomous Code Generation System

Un sistema autonomo di sviluppo software basato su architettura **Planner-Executor** che utilizza modelli LLM locali per generare codice seguendo metodologia TDD (Test Driven Development).

## 📋 Indice

- [Panoramica](#panoramica)
- [Architettura](#architettura)
- [Caratteristiche](#caratteristiche)
- [Requisiti](#requisiti)
- [Installazione](#installazione)
- [Configurazione](#configurazione)
- [Utilizzo](#utilizzo)
- [Struttura del Progetto](#struttura-del-progetto)
- [Processo di Sviluppo](#processo-di-sviluppo)
- [Workflow delle Feature](#workflow-delle-feature)
- [Documentazione Generata](#documentazione-generata)
- [Gestione Git](#gestione-git)
- [Troubleshooting](#troubleshooting)

## 🎯 Panoramica

L'**AI Development Agent** è un sistema autonomo che:

- **Pianifica** lo sviluppo usando un modello LLM dedicato (Planner)
- **Genera codice** usando un modello LLM specializzato (Executor)
- **Segue TDD** rigorosamente: Test → Codice → Refactor
- **Gestisce Git** automaticamente: crea repository e committa ogni feature
- **Genera documentazione** per ogni feature e documento finale completo
- **Cicla fino a correzione** di tutti gli errori prima di procedere

## 🏗️ Architettura

### Planner-Executor Pattern

Il sistema utilizza due modelli LLM distinti:

#### 1. **Planner (Qwen2.5-7B-Instruct)**
- **Ruolo**: Architetto software senior
- **Responsabilità**:
  - Analizza il task in `input/task.txt`
  - Pianifica lo sviluppo feature per feature
  - Genera piani JSON con azioni specifiche
  - Gestisce il workflow TDD
  - Coordina test e regression test
  - Decide quando committare su Git

#### 2. **Executor (Qwen2.5-Coder-32B-Instruct)**
- **Ruolo**: Sviluppatore esperto
- **Responsabilità**:
  - Riceve istruzioni dettagliate dal Planner
  - Genera codice Python puro (no markdown, no spiegazioni)
  - Scrive file seguendo le specifiche
  - Focus esclusivo sulla scrittura del codice

### Flusso di Comunicazione

```
Task (input/task.txt)
    ↓
Planner → Analizza → Genera Piano JSON
    ↓
Executor → Riceve Istruzioni → Genera Codice
    ↓
ToolManager → Esegue Test → Feedback
    ↓
Planner → Valuta Risultati → Prossima Azione
```

## ✨ Caratteristiche

### 🎯 Sviluppo Feature-by-Feature
- Una feature alla volta
- Ogni feature deve essere completa (codice + test + documentazione + commit) prima della successiva

### 🧪 Test Driven Development (TDD)
- **Red**: Scrive il test (fallisce)
- **Green**: Scrive il codice (test passa)
- **Refactor**: Migliora il codice
- **Regression**: Esegue tutti i test esistenti

### 📚 Documentazione Automatica
- Documentazione per ogni feature in `output/docs/features/`
- Documento finale `output/README.md` con:
  - Overview del progetto
  - Istruzioni di compilazione
  - Istruzioni di esecuzione
  - Guida al deployment in produzione

### 🔄 Gestione Errori Robusta
- **Cicla indefinitamente** fino a correzione di tutti gli errori
- Non procede se i test falliscono
- Feedback dettagliato su ogni errore

### 🔧 Gestione Git Automatica
- Crea repository Git in `output/` se non esiste
- Rileva repository esistente (non lo ricrea)
- Commit automatico per ogni feature completata
- Supporto push remoto con token

### 📁 Organizzazione Output
- Tutto il codice generato in `output/`
- Struttura organizzata automaticamente
- Repository Git separato in `output/.git/`

## 📦 Requisiti

### Software
- **Python 3.10+**
- **pip** (Python package manager)
- **Git** (per gestione repository)

### Hardware
- **Server LLM locale** con due modelli:
  - Planner: Qwen2.5-7B-Instruct (o equivalente)
  - Executor: Qwen2.5-Coder-32B-Instruct (o equivalente)
- **RAM**: Minimo 16GB (consigliato 32GB+ per modelli grandi)
- **GPU**: Consigliata per performance migliori

### Server LLM
Il sistema richiede due server LLM locali che espongono API compatibili con OpenAI:
- **Planner Server**: Porta 8081 (default)
- **Executor Server**: Porta 8080 (default)

Utilizza [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) o server compatibili.

## 🚀 Installazione

### 1. Clona o scarica il progetto

```bash
git clone <repository-url>
cd LongRunDualDevAgent
```

### 2. Crea ambiente virtuale

```bash
python3 -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
```

### 3. Installa dipendenze

```bash
pip install -r requirements.txt
```

### 4. Configura il progetto

```bash
# Copia il file di esempio
cp config.json.example config.json

# Modifica config.json con le tue configurazioni
nano config.json  # o usa il tuo editor preferito
```

## ⚙️ Configurazione

### File `config.json`

Il file di configurazione contiene tutte le impostazioni necessarie:

```json
{
  "git_token": "YOUR_GIT_TOKEN_HERE",
  "api_key": "ALTERNATIVE_API_KEY_FIELD",
  "planner": {
    "server": "http://192.168.1.29:8081",
    "model": "bartowski_Qwen2.5-7B-Instruct-GGUF_Qwen2.5-7B-Instruct-Q4_K_S.gguf",
    "timeout": 120,
    "temperature": 0.7
  },
  "executor": {
    "server": "http://192.168.1.29:8080",
    "model": "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
    "timeout": 240,
    "temperature": 0.2
  }
}
```

### Parametri di Configurazione

#### Planner
- **server**: URL del server LLM per il Planner
- **model**: Nome del modello da utilizzare
- **timeout**: Timeout in secondi (default: 120)
- **temperature**: Temperatura di sampling (0.7 per planning)

#### Executor
- **server**: URL del server LLM per l'Executor
- **model**: Nome del modello da utilizzare
- **timeout**: Timeout in secondi (default: 240 per modelli grandi)
- **temperature**: Temperatura di sampling (0.2 per codice deterministico)

#### Git
- **git_token**: Token Git per push remoti (opzionale)
- **api_key**: Campo alternativo per token API

### Configurazione Server LLM

Assicurati che i server LLM siano in esecuzione e accessibili agli URL configurati.

Esempio con llama-cpp-python:
```bash
# Server Planner (porta 8081)
python -m llama_cpp.server --model path/to/planner-model.gguf --port 8081

# Server Executor (porta 8080)
python -m llama_cpp.server --model path/to/executor-model.gguf --port 8080
```

## 💻 Utilizzo

### 1. Prepara il Task

Crea o modifica il file `input/task.txt` con la descrizione del software da sviluppare:

```
Genera un software per la gestione completa del ristorante.
Dalla cucina agli ordini, al menù, al pagamento.
Il progetto deve essere web e deve avere interfacce utente
per ogni utente (cameriere, cuoco, receptionist, cassa).
```

### 2. Avvia l'Agente

```bash
# Metodo 1: Usa lo script di avvio (consigliato)
./run_agent.sh

# Metodo 2: Attiva venv e esegui manualmente
source venv/bin/activate
python3 code_agent.py
```

### 3. Monitora l'Esecuzione

L'agente mostrerà:
- 🤖 **Planner thinking...** - Il planner sta analizzando
- ✍️ **Executor coding...** - L'executor sta generando codice
- ⚙️ **Executing...** - Esecuzione comandi/test
- ✅ **Success** - Operazione completata
- ❌ **Error** - Errore rilevato (l'agente ciclerà per correggerlo)

### 4. Risultati

Tutto il codice generato sarà in:
- **`output/`** - Directory principale
- **`output/docs/features/`** - Documentazione feature
- **`output/README.md`** - Documentazione finale del progetto
- **`output/.git/`** - Repository Git del progetto generato

## 📂 Struttura del Progetto

```
LongRunDualDevAgent/
├── code_agent.py              # Script principale dell'agente
├── config.json                # Configurazione (NON committare!)
├── config.json.example         # Esempio configurazione
├── requirements.txt           # Dipendenze Python
├── run_agent.sh              # Script di avvio
├── .gitignore                 # File da escludere da Git
├── README.md                  # Questo file
│
├── input/                     # Input dell'agente
│   └── task.txt               # Descrizione del task da sviluppare
│
├── output/                    # Output generato (NON committare!)
│   ├── .git/                  # Repository Git del progetto generato
│   ├── README.md              # Documentazione finale del progetto
│   ├── docs/
│   │   └── features/          # Documentazione per feature
│   ├── src/                   # Codice sorgente generato
│   ├── tests/                 # Test generati
│   └── ...                    # Altri file del progetto
│
└── venv/                      # Ambiente virtuale Python (NON committare!)
```

## 🔄 Processo di Sviluppo

### Fase 1: Pianificazione
1. L'agente legge `input/task.txt`
2. Il **Planner** analizza il task
3. Il Planner genera un piano JSON con azioni specifiche
4. Il piano include: feature da sviluppare, file da creare, test da scrivere

### Fase 2: Sviluppo Feature
Per ogni feature:

1. **Scrittura Test (Red)**
   - Planner decide quali test scrivere
   - Executor genera il codice del test
   - Test viene salvato in `output/tests/`

2. **Scrittura Codice (Green)**
   - Planner decide l'implementazione
   - Executor genera il codice
   - Codice viene salvato in `output/src/`

3. **Esecuzione Test Feature**
   - ToolManager esegue il test specifico della feature
   - Se fallisce → Planner genera fix → ciclo fino a successo

4. **Esecuzione Regression Test**
   - ToolManager esegue l'intera suite di test
   - Verifica che nessuna funzionalità esistente sia rotta
   - Se fallisce → Planner genera fix → ciclo fino a successo

5. **Generazione Documentazione**
   - Sistema genera documentazione della feature
   - Salva in `output/docs/features/[feature_name].md`

6. **Commit Git**
   - Solo se tutti i test passano
   - Commit message: `"Feature: [name] - implemented and tested"`
   - Push opzionale se token configurato

### Fase 3: Feature Successiva
- Solo dopo completamento completo della feature precedente
- Processo si ripete per ogni feature

### Fase 4: Finalizzazione
- Quando tutte le feature sono complete
- Genera `output/README.md` finale
- Commit finale del progetto

## 🎯 Workflow delle Feature

```
┌─────────────────────────────────────────┐
│  Planner: Identifica Feature            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Planner: Genera Piano (JSON)            │
│  - write_file: test_feature.py          │
│  - write_file: feature.py                │
│  - execute_command: pytest test_feature  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Executor: Genera Codice Test           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Executor: Genera Codice Feature        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  ToolManager: Esegue Test Feature       │
└──────────────┬──────────────────────────┘
               │
         ┌─────┴─────┐
         │  Passa?   │
         └─────┬─────┘
               │ NO
               │ ┌──────────────────────┐
               └─┤ Planner: Genera Fix   │
                 │ Executor: Corregge    │
                 │ ToolManager: Ritesa   │
                 └───────────┬──────────┘
                             │
                             └───► Cicla fino a successo
               │ YES
               ▼
┌─────────────────────────────────────────┐
│  ToolManager: Esegue Regression Test   │
└──────────────┬──────────────────────────┘
               │
         ┌─────┴─────┐
         │  Passa?   │
         └─────┬─────┘
               │ NO
               │ ┌──────────────────────┐
               └─┤ Planner: Genera Fix   │
                 │ Executor: Corregge    │
                 │ ToolManager: Ritesa   │
                 └───────────┬──────────┘
                             │
                             └───► Cicla fino a successo
               │ YES
               ▼
┌─────────────────────────────────────────┐
│  Sistema: Genera Documentazione         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Git: Commit Feature                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Planner: Prossima Feature             │
└─────────────────────────────────────────┘
```

## 📚 Documentazione Generata

### Documentazione Feature

Per ogni feature completata, viene generato un file in `output/docs/features/[feature_name].md` contenente:
- Overview della feature
- File di implementazione
- File di test
- Stato di completamento

### Documentazione Finale

Il file `output/README.md` finale include:
- **Overview del progetto**: Descrizione generale
- **Lista delle feature**: Tutte le feature implementate
- **Struttura del progetto**: Organizzazione dei file
- **Building**: Come compilare il progetto
- **Running**: Come eseguire il progetto
- **Deployment**: Guida al deployment in produzione

## 🔧 Gestione Git

### Repository Automatico

L'agente gestisce automaticamente il repository Git:

1. **Creazione Repository**
   - Verifica se esiste `output/.git/`
   - Se non esiste, crea nuovo repository
   - Configura user.name e user.email automaticamente

2. **Commit Automatici**
   - Un commit per ogni feature completata
   - Solo se tutti i test passano
   - Message: `"Feature: [name] - implemented and tested"`

3. **Push Remoto (Opzionale)**
   - Se `git_token` è configurato
   - Push automatico dopo ogni commit
   - Supporta GitHub e GitLab

### Configurazione Git Token

Per abilitare push remoti:

1. Genera un token Git (GitHub/GitLab)
2. Aggiungi al `config.json`:
   ```json
   {
     "git_token": "ghp_xxxxxxxxxxxxxxxxxxxx"
   }
   ```

## 🐛 Troubleshooting

### Errore: "ModuleNotFoundError: No module named 'requests'"

**Soluzione**: Installa le dipendenze
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Errore: "Connection error" o "Request timeout"

**Possibili cause**:
- Server LLM non in esecuzione
- URL o porta errati in `config.json`
- Timeout troppo basso per modelli grandi

**Soluzione**:
1. Verifica che i server LLM siano attivi
2. Controlla URL e porte in `config.json`
3. Aumenta `timeout` se necessario

### L'agente non genera codice

**Possibili cause**:
- Planner non riceve risposte valide
- JSON malformato dal Planner
- Errori di parsing

**Soluzione**:
1. Controlla i log per errori specifici
2. Verifica che i modelli siano correttamente configurati
3. Aumenta `temperature` del Planner se necessario

### Test falliscono continuamente

**Comportamento atteso**: L'agente cicla fino a correzione. Se continua a fallire:
1. Verifica che il task in `input/task.txt` sia chiaro
2. Controlla i log per capire cosa sta fallendo
3. Potrebbe essere necessario migliorare i prompt del Planner

### Repository Git non viene creato

**Soluzione**:
1. Verifica permessi di scrittura in `output/`
2. Controlla che Git sia installato
3. L'agente crea il repository al primo file generato

## 📝 Note Importanti

### File da NON Committare

- `config.json` - Contiene token sensibili
- `output/` - Ha il suo repository Git separato
- `venv/` - Ambiente virtuale
- File temporanei e cache

### Sicurezza

- **NON committare** `config.json` con token reali
- Usa `config.json.example` come template
- Il file è già in `.gitignore`

### Performance

- Modelli grandi (32B+) richiedono molto RAM/VRAM
- Timeout elevati per modelli grandi
- Considera GPU per performance migliori

## 🤝 Contribuire

Per contribuire al progetto:

1. Fork del repository
2. Crea branch per feature (`git checkout -b feature/AmazingFeature`)
3. Commit delle modifiche (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## 📄 Licenza

[Specifica la licenza del progetto]

## 👤 Autore

[Informazioni sull'autore]

---

**Sviluppato con ❤️ usando architettura Planner-Executor e TDD**
