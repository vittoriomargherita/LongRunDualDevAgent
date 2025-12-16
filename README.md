# AI Development Agent - Autonomous Code Generation System

An autonomous software development system based on a **Planner-Executor** architecture that uses local LLM models to generate code following TDD (Test Driven Development) methodology.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Development Process](#development-process)
- [Feature Workflow](#feature-workflow)
- [Generated Documentation](#generated-documentation)
- [Git Management](#git-management)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The **AI Development Agent** is an autonomous system that:

- **Plans** development using a dedicated LLM model (Planner)
- **Generates code** using a specialized LLM model (Executor)
- **Follows TDD** rigorously: Test → Code → Refactor
- **Manages Git** automatically: creates repository and commits each feature
- **Generates documentation** for each feature and complete final document
- **Cycles until correction** of all errors before proceeding

## 🏗️ Architecture

### Planner-Executor Pattern

The system uses two distinct LLM models:

#### 1. **Planner (Qwen2.5-7B-Instruct)**
- **Role**: Senior software architect
- **Responsibilities**:
  - Analyzes the task in `input/task.txt`
  - Plans development feature by feature
  - Generates JSON plans with specific actions
  - Manages TDD workflow
  - Coordinates tests and regression tests
  - Decides when to commit to Git

#### 2. **Executor (Qwen2.5-Coder-32B-Instruct)**
- **Role**: Expert developer
- **Responsibilities**:
  - Receives detailed instructions from the Planner
  - Generates pure Python code (no markdown, no explanations)
  - Writes files following specifications
  - Exclusive focus on code writing

### Communication Flow

```
Task (input/task.txt)
    ↓
Planner → Analyzes → Generates JSON Plan
    ↓
Executor → Receives Instructions → Generates Code
    ↓
ToolManager → Executes Tests → Feedback
    ↓
Planner → Evaluates Results → Next Action
```

## ✨ Features

### 🎯 Feature-by-Feature Development
- One feature at a time
- Each feature must be complete (code + tests + documentation + commit) before the next one

### 🧪 Test Driven Development (TDD)
- **Red**: Writes the test (fails)
- **Green**: Writes the code (test passes)
- **Refactor**: Improves the code
- **Regression**: Executes all existing tests

### 📚 Automatic Documentation
- Documentation for each feature in `output/docs/features/`
- Final document `output/README.md` with:
  - Project overview
  - Build instructions
  - Execution instructions
  - Production deployment guide

### 🔄 Robust Error Handling
- **Cycles indefinitely** until all errors are corrected
- Does not proceed if tests fail
- Detailed feedback on each error

### 🔧 Automatic Git Management
- Creates Git repository in `output/` if it doesn't exist
- Detects existing repository (doesn't recreate it)
- Automatic commit for each completed feature
- Remote push support with token

### 📁 Output Organization
- All generated code in `output/`
- Automatically organized structure
- Separate Git repository in `output/.git/`

## 📦 Requirements

### Software
- **Python 3.10+**
- **pip** (Python package manager)
- **Git** (for repository management)

### Hardware
- **Local LLM server** with two models:
  - Planner: Qwen2.5-7B-Instruct (or equivalent)
  - Executor: Qwen2.5-Coder-32B-Instruct (or equivalent)
- **RAM**: Minimum 16GB (32GB+ recommended for large models)
- **GPU**: Recommended for better performance

### LLM Server
The system requires two local LLM servers that expose OpenAI-compatible APIs:
- **Planner Server**: Port 8081 (default)
- **Executor Server**: Port 8080 (default)

Uses [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) or compatible servers.

## 🚀 Installation

### 1. Clone or download the project

```bash
git clone https://github.com/vittoriomargherita/LongRunDualDevAgent.git
cd LongRunDualDevAgent
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the project

```bash
# Copy the example file
cp config.json.example config.json

# Edit config.json with your configurations
nano config.json  # or use your preferred editor
```

## ⚙️ Configuration

### `config.json` File

The configuration file contains all necessary settings:

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

### Configuration Parameters

#### Planner
- **server**: URL of the LLM server for the Planner
- **model**: Name of the model to use
- **timeout**: Timeout in seconds (default: 120)
- **temperature**: Sampling temperature (0.7 for planning)

#### Executor
- **server**: URL of the LLM server for the Executor
- **model**: Name of the model to use
- **timeout**: Timeout in seconds (default: 240 for large models)
- **temperature**: Sampling temperature (0.2 for deterministic code)

#### Git
- **git_token**: Git token for remote push (optional)
- **api_key**: Alternative field for API token

### LLM Server Configuration

Make sure the LLM servers are running and accessible at the configured URLs.

Example with llama-cpp-python:
```bash
# Planner Server (port 8081)
python -m llama_cpp.server --model path/to/planner-model.gguf --port 8081

# Executor Server (port 8080)
python -m llama_cpp.server --model path/to/executor-model.gguf --port 8080
```

## 💻 Usage

### 1. Prepare the Task

Create or edit the `input/task.txt` file with the description of the software to develop:

```
Generate a complete restaurant management software.
From kitchen to orders, menu, to payment.
The project must be web-based and must have user interfaces
for each user (waiter, cook, receptionist, cashier).
```

### 2. Start the Agent

```bash
# Method 1: Use the startup script (recommended)
./run_agent.sh

# Method 2: Activate venv and run manually
source venv/bin/activate
python3 code_agent.py
```

### 3. Monitor Execution

The agent will show:
- 🤖 **Planner thinking...** - The planner is analyzing
- ✍️ **Executor coding...** - The executor is generating code
- ⚙️ **Executing...** - Executing commands/tests
- ✅ **Success** - Operation completed
- ❌ **Error** - Error detected (the agent will cycle to correct it)

### 4. Results

All generated code will be in:
- **`output/`** - Main directory
- **`output/docs/features/`** - Feature documentation
- **`output/README.md`** - Final project documentation
- **`output/.git/`** - Git repository of the generated project

## 📂 Project Structure

```
LongRunDualDevAgent/
├── code_agent.py              # Main agent script
├── config.json                # Configuration (DO NOT commit!)
├── config.json.example         # Configuration example
├── requirements.txt           # Python dependencies
├── run_agent.sh              # Startup script
├── .gitignore                 # Files to exclude from Git
├── README.md                  # This file
│
├── input/                     # Agent input
│   └── task.txt               # Description of the task to develop
│
├── output/                    # Generated output (DO NOT commit!)
│   ├── .git/                  # Git repository of the generated project
│   ├── README.md              # Final project documentation
│   ├── docs/
│   │   └── features/          # Feature documentation
│   ├── src/                   # Generated source code
│   ├── tests/                 # Generated tests
│   └── ...                    # Other project files
│
└── venv/                      # Python virtual environment (DO NOT commit!)
```

## 🔄 Development Process

### Phase 1: Planning
1. The agent reads `input/task.txt`
2. The **Planner** analyzes the task
3. The Planner generates a JSON plan with specific actions
4. The plan includes: features to develop, files to create, tests to write

### Phase 2: Feature Development
For each feature:

1. **Test Writing (Red)**
   - Planner decides which tests to write
   - Executor generates the test code
   - Test is saved in `output/tests/`

2. **Code Writing (Green)**
   - Planner decides the implementation
   - Executor generates the code
   - Code is saved in `output/src/`

3. **Feature Test Execution**
   - ToolManager executes the feature-specific test
   - If it fails → Planner generates fix → cycle until success

4. **Regression Test Execution**
   - ToolManager executes the entire test suite
   - Verifies that no existing functionality is broken
   - If it fails → Planner generates fix → cycle until success

5. **Documentation Generation**
   - System generates feature documentation
   - Saves in `output/docs/features/[feature_name].md`

6. **Git Commit**
   - Only if all tests pass
   - Commit message: `"Feature: [name] - implemented and tested"`
   - Optional push if token is configured

### Phase 3: Next Feature
- Only after complete completion of the previous feature
- Process repeats for each feature

### Phase 4: Finalization
- When all features are complete
- Generates final `output/README.md`
- Final project commit

## 🎯 Feature Workflow

```
┌─────────────────────────────────────────┐
│  Planner: Identifies Feature            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Planner: Generates Plan (JSON)          │
│  - write_file: test_feature.py          │
│  - write_file: feature.py                │
│  - execute_command: pytest test_feature  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Executor: Generates Test Code          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Executor: Generates Feature Code       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  ToolManager: Executes Feature Test     │
└──────────────┬──────────────────────────┘
               │
         ┌─────┴─────┐
         │  Pass?    │
         └─────┬─────┘
               │ NO
               │ ┌──────────────────────┐
               └─┤ Planner: Generates Fix│
                 │ Executor: Corrects   │
                 │ ToolManager: Retries  │
                 └───────────┬──────────┘
                             │
                             └───► Cycles until success
               │ YES
               ▼
┌─────────────────────────────────────────┐
│  ToolManager: Executes Regression Test  │
└──────────────┬──────────────────────────┘
               │
         ┌─────┴─────┐
         │  Pass?    │
         └─────┬─────┘
               │ NO
               │ ┌──────────────────────┐
               └─┤ Planner: Generates Fix│
                 │ Executor: Corrects   │
                 │ ToolManager: Retries  │
                 └───────────┬──────────┘
                             │
                             └───► Cycles until success
               │ YES
               ▼
┌─────────────────────────────────────────┐
│  System: Generates Documentation        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Git: Commits Feature                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Planner: Next Feature                  │
└─────────────────────────────────────────┘
```

## 📚 Generated Documentation

### Feature Documentation

For each completed feature, a file is generated in `output/docs/features/[feature_name].md` containing:
- Feature overview
- Implementation files
- Test files
- Completion status

### Final Documentation

The final `output/README.md` file includes:
- **Project Overview**: General description
- **Feature List**: All implemented features
- **Project Structure**: File organization
- **Building**: How to build the project
- **Running**: How to run the project
- **Deployment**: Production deployment guide

## 🔧 Git Management

### Automatic Repository

The agent automatically manages the Git repository:

1. **Repository Creation**
   - Checks if `output/.git/` exists
   - If it doesn't exist, creates new repository
   - Automatically configures user.name and user.email

2. **Automatic Commits**
   - One commit for each completed feature
   - Only if all tests pass
   - Message: `"Feature: [name] - implemented and tested"`

3. **Remote Push (Optional)**
   - If `git_token` is configured
   - Automatic push after each commit
   - Supports GitHub and GitLab

### Git Token Configuration

To enable remote push:

1. Generate a Git token (GitHub/GitLab)
2. Add to `config.json`:
   ```json
   {
     "git_token": "ghp_xxxxxxxxxxxxxxxxxxxx"
   }
   ```

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'requests'"

**Solution**: Install dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Error: "Connection error" or "Request timeout"

**Possible causes**:
- LLM server not running
- Incorrect URL or port in `config.json`
- Timeout too low for large models

**Solution**:
1. Verify that LLM servers are active
2. Check URL and ports in `config.json`
3. Increase `timeout` if necessary

### Agent doesn't generate code

**Possible causes**:
- Planner doesn't receive valid responses
- Malformed JSON from Planner
- Parsing errors

**Solution**:
1. Check logs for specific errors
2. Verify that models are correctly configured
3. Increase Planner `temperature` if necessary

### Tests fail continuously

**Expected behavior**: The agent cycles until correction. If it continues to fail:
1. Verify that the task in `input/task.txt` is clear
2. Check logs to understand what is failing
3. It may be necessary to improve Planner prompts

### Git repository is not created

**Solution**:
1. Verify write permissions in `output/`
2. Check that Git is installed
3. The agent creates the repository at the first generated file

## 📝 Important Notes

### Files to NOT Commit

- `config.json` - Contains sensitive tokens
- `output/` - Has its own separate Git repository
- `venv/` - Virtual environment
- Temporary files and cache

### Security

- **DO NOT commit** `config.json` with real tokens
- Use `config.json.example` as a template
- The file is already in `.gitignore`

### Performance

- Large models (32B+) require a lot of RAM/VRAM
- High timeouts for large models
- Consider GPU for better performance

## 🤝 Contributing

To contribute to the project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

[Specify project license]

## 👤 Author

[Author information]

---

**Developed with ❤️ using Planner-Executor architecture and TDD**
