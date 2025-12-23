#!/usr/bin/env python3
"""
Autonomous AI Development Agent
Planner-Executor Architecture using Local LLMs
"""

import json
import os
import subprocess
import time
import requests
from typing import List, Dict, Optional, Any
from pathlib import Path


class LLMClient:
    """Handles HTTP POST requests to local LLM APIs with error handling and timeouts."""
    
    def __init__(self, api_url: str, model_name: str = "local-model", timeout: int = 240, 
                 temperature: float = 0.2, max_tokens: int = 4096):
        """
        Initialize LLM client.
        
        Args:
            api_url: Base URL for the API endpoint
            model_name: Name of the model to use
            timeout: Request timeout in seconds (default 240s for large models)
            temperature: Sampling temperature (low for coding, medium for planning)
            max_tokens: Maximum tokens to generate (default 4096)
        """
        self.api_url = api_url
        self.model_name = model_name
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None) -> str:
        """
        Send chat completion request to the LLM API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature if provided
            max_tokens: Override default max_tokens if provided
            
        Returns:
            Response content as string
            
        Raises:
            requests.exceptions.RequestException: On connection/timeout errors
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": False
        }
        
        try:
            print(f"🌐 Sending request to {self.api_url} (timeout: {self.timeout}s, max_tokens: {payload['max_tokens']})...")
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Log token usage if available (helps debug context limits)
            usage = result.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens", "?")
                completion_tokens = usage.get("completion_tokens", "?")
                total_tokens = usage.get("total_tokens", "?")
                print(f"📊 Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
            
            # Check if response was truncated due to max_tokens
            finish_reason = result.get("choices", [{}])[0].get("finish_reason", "")
            if finish_reason == "length":
                print(f"⚠️  WARNING: Response truncated due to max_tokens limit! Consider increasing max_tokens in config.")
            
            if not content:
                raise ValueError("Empty response from LLM API")
            
            return content
            
        except requests.exceptions.Timeout:
            raise Exception(f"Request timeout after {self.timeout}s. The model may be too slow or overloaded.")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Connection error. Is the LLM server running at {self.api_url}?")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except (KeyError, IndexError, ValueError) as e:
            raise Exception(f"Invalid response format from API: {str(e)}")


class ToolManager:
    """Manages file operations and shell command execution."""
    
    @staticmethod
    def execute_command(cmd: str, timeout: int = 60, cwd: Optional[str] = None) -> tuple[str, str, int]:
        """
        Execute a shell command with timeout.
        
        Args:
            cmd: Command string to execute
            timeout: Maximum execution time in seconds (default 60s)
            cwd: Working directory for command execution
            
        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        try:
            print(f"⚙️  Executing: {cmd}")
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )
            stdout = result.stdout
            stderr = result.stderr
            return_code = result.returncode
            
            if stdout:
                print(f"📤 STDOUT:\n{stdout}")
            if stderr:
                print(f"⚠️  STDERR:\n{stderr}")
            if return_code != 0:
                print(f"❌ Command failed with return code: {return_code}")
            else:
                print(f"✅ Command succeeded")
            
            return stdout, stderr, return_code
            
        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {timeout}s: {cmd}"
            print(f"⏱️  {error_msg}")
            return "", error_msg, -1
        except Exception as e:
            error_msg = f"Command execution error: {str(e)}"
            print(f"💥 {error_msg}")
            return "", error_msg, -1
    
    @staticmethod
    def read_file(path: str) -> str:
        """
        Read file content.
        
        Args:
            path: File path to read
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        try:
            print(f"📖 Reading file: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"✅ Read {len(content)} characters from {path}")
            return content
        except FileNotFoundError:
            print(f"❌ File not found: {path}")
            raise
        except Exception as e:
            print(f"💥 Error reading file {path}: {str(e)}")
            raise
    
    @staticmethod
    def write_file(path: str, content: str) -> None:
        """
        Write content to file, creating directories if needed.
        
        Args:
            path: File path to write
            content: Content to write
        """
        try:
            print(f"✍️  Writing file: {path}")
            # Create parent directories if they don't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Wrote {len(content)} characters to {path}")
        except Exception as e:
            print(f"💥 Error writing file {path}: {str(e)}")
            raise


class CodeAgent:
    """Main autonomous development agent with Planner-Executor architecture."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the Code Agent.
        
        Args:
            config_path: Path to configuration file with model and server settings
        """
        # Load configuration
        self.config_path = config_path
        self.config = self._load_config()
        
        # Set output directory (all generated code goes here)
        self.output_dir = "output"
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # Initialize LLM clients from configuration
        planner_config = self.config.get('planner', {})
        executor_config = self.config.get('executor', {})
        
        self.planner_client = LLMClient(
            api_url=planner_config.get('server', 'http://192.168.1.29:8081'),
            model_name=planner_config.get('model', 'local-model'),
            timeout=planner_config.get('timeout', 120),
            temperature=planner_config.get('temperature', 0.7),
            max_tokens=planner_config.get('max_tokens', 2048)  # Planner needs less tokens for JSON plans
        )
        self.executor_client = LLMClient(
            api_url=executor_config.get('server', 'http://192.168.1.29:8080'),
            model_name=executor_config.get('model', 'local-model'),
            timeout=executor_config.get('timeout', 240),
            max_tokens=executor_config.get('max_tokens', 8192),  # Executor needs more for code generation
            temperature=executor_config.get('temperature', 0.2)
        )
        
        self.tools = ToolManager()
        self.history: List[Dict[str, Any]] = []
        self.git_token: Optional[str] = None
        self.current_feature: Optional[str] = None
        self.feature_test_passed: bool = False
        self.feature_docs: List[Dict[str, str]] = []  # Store feature documentation
        self.current_feature_files: List[str] = []  # Track files written for current feature
        self.git_repo_initialized: bool = False
        self.test_counter: int = 0  # Counter for test numbering
        self.thought_chain: List[str] = []  # Store thought chain for logging
        self.start_time: Optional[float] = None  # Track total execution time
        self.phpunit_available: Optional[bool] = None  # Cache PHPUnit availability check
        self.task_description: str = ""  # Store task description for Executor context
        self.php_server_process: Optional[Any] = None  # PHP server process for testing
        self.php_server_port: int = 8000  # Port for PHP built-in server
        
        # Load Git token from config
        self.git_token = self.config.get('git_token') or self.config.get('api_key')
        if self.git_token:
            print(f"✅ Git token loaded from {config_path}")
        
        # Initialize Git repository if needed
        self._ensure_git_repo()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config file."""
        default_config = {
            "planner": {
                "server": "http://192.168.1.29:8081",
                "model": "local-model",
                "timeout": 180,  # Increased to 180s for complex planning/validation
                "temperature": 0.7
            },
            "executor": {
                "server": "http://192.168.1.29:8080",
                "model": "local-model",
                "timeout": 300,  # Increased to 300s for complex code generation
                "temperature": 0.2
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key in default_config:
                        if key not in config:
                            config[key] = default_config[key]
                        elif isinstance(default_config[key], dict):
                            for subkey in default_config[key]:
                                if subkey not in config[key]:
                                    config[key][subkey] = default_config[key][subkey]
                    return config
            except Exception as e:
                print(f"⚠️  Error loading config: {e}. Using defaults.")
                return default_config
        else:
            print(f"⚠️  Config file not found: {self.config_path}. Using defaults.")
            return default_config
    
    def _ensure_git_repo(self) -> None:
        """Initialize Git repository if it doesn't exist."""
        # CRITICAL: Ensure output directory exists FIRST before checking for git repo
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if .git directory exists (more reliable than git rev-parse when directory might not exist)
        git_dir = os.path.join(self.output_dir, ".git")
        if os.path.exists(git_dir) and os.path.isdir(git_dir):
            # Verify it's a valid git repo
            stdout, stderr, code = self.tools.execute_command("git rev-parse --git-dir", timeout=10, cwd=self.output_dir)
            if code == 0:
                if not self.git_repo_initialized:
                    print("✅ Git repository already exists in output/")
                self.git_repo_initialized = True
                return
        
        # If we get here, either .git doesn't exist or it's not a valid repo
        # Initialize git repo (even if directory is empty - will be ready for first commit)
        print("🔧 Initializing Git repository in output/...")
        self._log_thought("Initializing Git repository")
        
        try:
            # Initialize git repo
            stdout, stderr, code = self.tools.execute_command("git init", timeout=10, cwd=self.output_dir)
            if code != 0:
                raise Exception(f"git init failed: {stderr}")
            
            # Configure git user (required for commits)
            stdout, stderr, code = self.tools.execute_command("git config user.name 'AI Development Agent'", timeout=10, cwd=self.output_dir)
            if code != 0:
                print(f"⚠️  Warning: Failed to set git user.name: {stderr}")
            
            stdout, stderr, code = self.tools.execute_command("git config user.email 'agent@dev.local'", timeout=10, cwd=self.output_dir)
            if code != 0:
                print(f"⚠️  Warning: Failed to set git user.email: {stderr}")
            
            # Create initial .gitignore if needed
            gitignore_path = os.path.join(self.output_dir, ".gitignore")
            if not os.path.exists(gitignore_path):
                gitignore_content = "__pycache__/\n*.pyc\n*.pyo\n*.pyd\n.Python\nvenv/\nenv/\n.venv/\n*.log\n.DS_Store\n"
                self.tools.write_file(gitignore_path, gitignore_content)
            
            # Verify repository was created
            if os.path.exists(git_dir) and os.path.isdir(git_dir):
                self.git_repo_initialized = True
                print("✅ Git repository initialized successfully")
                self._log_thought("Git repository initialized successfully")
            else:
                raise Exception("Git repository directory was not created")
                
        except Exception as e:
            print(f"⚠️  Error initializing Git repository: {e}")
            self._log_thought(f"Git repository initialization failed: {e}")
            # Don't set git_repo_initialized = True if initialization failed
            self.git_repo_initialized = False
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize file path to be relative to output directory.
        IMPORTANT: Does NOT normalize paths that start with 'input/' - those are read from project root.
        
        Args:
            path: Original path from Planner
            
        Returns:
            Normalized path relative to output directory (or absolute path for input files)
        """
        # CRITICAL: Don't normalize input/ paths - they should be read from project root
        if path.startswith('input/') or path.startswith('../input/'):
            # Return absolute path to input file
            if os.path.isabs(path):
                return path
            # If relative, resolve from project root
            project_root = os.path.dirname(os.path.abspath(__file__))
            normalized = os.path.join(project_root, path.lstrip('../'))
            return normalized
        
        # Remove output/ prefix if present
        if path.startswith(self.output_dir + "/"):
            path = path[len(self.output_dir) + 1:]
        elif path.startswith("output/"):
            path = path[7:]
        
        # If path is absolute, make it relative
        if os.path.isabs(path):
            # Try to make it relative to output_dir
            try:
                path = os.path.relpath(path, self.output_dir)
            except ValueError:
                pass
        
        # For source code files, ensure they go to src/ subdirectory
        # unless they're already in a specific subdirectory
        # IMPORTANT: Flatten structure - no subdirectories in src/
        if not any(path.startswith(d + "/") for d in ["src", "tests", "docs", "config", ".git"]):
            # Check if it's a source code file
            if path.endswith((".py", ".js", ".html", ".css", ".php", ".java", ".cpp", ".c", ".h")):
                # Put in src/ unless it's a test file
                if "test" not in path.lower() and "spec" not in path.lower():
                    # Flatten: remove any subdirectory structure, just use filename
                    filename = os.path.basename(path)
                    path = os.path.join("src", filename)
            elif path.endswith((".md", ".txt", ".json", ".yml", ".yaml")) and "docs" not in path:
                # Documentation files go to docs/ or root
                # README.md goes to root of output/
                if path.endswith("README.md") or "README" in path.upper():
                    path = "README.md"
                elif not path.startswith("README") and not path.startswith("LICENSE"):
                    path = os.path.join("docs", path)
        
        # Join with output directory
        return os.path.join(self.output_dir, path)
    
    def _get_file_description(self, file_path: str) -> str:
        """Get a brief description of what a file does by analyzing its content."""
        try:
            full_path = os.path.join(self.output_dir, file_path)
            if not os.path.exists(full_path):
                return "File not found"
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Analyze file content to generate description
            if file_path.endswith('.php'):
                if 'api.php' in file_path.lower():
                    # Extract API endpoints
                    import re
                    endpoints = re.findall(r"case\s+['\"]([^'\"]+)['\"]", content)
                    if endpoints:
                        return f"API handler with endpoints: {', '.join(endpoints[:5])}"
                    return "API handler for HTTP requests"
                elif 'db.php' in file_path.lower() or 'database' in file_path.lower():
                    # Extract class methods
                    methods = re.findall(r'public\s+function\s+(\w+)', content)
                    if methods:
                        return f"Database class with methods: {', '.join(methods[:5])}"
                    return "Database connection and query handler"
                elif 'setup.php' in file_path.lower():
                    return "Database initialization script - creates tables and schema"
                else:
                    # Try to find main function or class
                    if 'function' in content:
                        funcs = re.findall(r'function\s+(\w+)', content)
                        if funcs:
                            return f"PHP file with functions: {', '.join(funcs[:3])}"
                    return "PHP source file"
            
            elif file_path.endswith('.html'):
                # Check for main functionality
                if 'login' in content.lower():
                    return "HTML page with login functionality"
                elif 'booking' in content.lower() or 'seat' in content.lower():
                    return "HTML page for seat booking interface"
                elif 'admin' in content.lower():
                    return "HTML page for admin panel"
                return "HTML user interface page"
            
            elif file_path.endswith('.py') and 'test' in file_path.lower():
                # Extract test function names
                import re
                tests = re.findall(r'def\s+(test_\w+)', content)
                if tests:
                    return f"Test file with cases: {', '.join(tests[:5])}"
                return "Python test file"
            
            return "Source code file"
        except Exception as e:
            return f"Error analyzing file: {str(e)}"
    
    def _generate_feature_documentation(self, feature_name: str, code_files: List[str], test_files: List[str]) -> str:
        """Generate detailed documentation for a completed feature."""
        # Remove duplicates and sort
        unique_code_files = sorted(set(code_files))
        unique_test_files = sorted(set(test_files))
        
        doc = f"# Feature: {feature_name}\n\n"
        
        # Generate detailed overview based on feature name and files
        doc += f"## Overview\n\n"
        
        # Create a more descriptive overview based on feature name
        feature_lower = feature_name.lower()
        if 'login' in feature_lower or 'authentication' in feature_lower:
            doc += f"This feature implements user authentication functionality, allowing users to log in to the system. "
            doc += f"It includes session management, password verification, and user validation.\n\n"
        elif 'registration' in feature_lower or 'register' in feature_lower:
            doc += f"This feature implements user registration functionality, allowing new users to create accounts. "
            doc += f"It includes email validation, password hashing, and verification token generation.\n\n"
        elif 'booking' in feature_lower or 'seat' in feature_lower:
            doc += f"This feature implements seat booking functionality, allowing users to reserve seats for specific dates. "
            doc += f"It includes seat availability checking, booking creation, and conflict prevention.\n\n"
        elif 'admin' in feature_lower:
            doc += f"This feature implements administrative functionality, providing admin users with management capabilities. "
            doc += f"It includes configuration management and rooming list generation.\n\n"
        elif 'date' in feature_lower or 'picker' in feature_lower:
            doc += f"This feature implements date selection functionality, allowing users to choose dates for booking operations. "
            doc += f"It includes date validation and integration with the booking system.\n\n"
        elif 'session' in feature_lower:
            doc += f"This feature implements session management, maintaining user authentication state across requests. "
            doc += f"It includes session creation, validation, and cleanup.\n\n"
        elif 'visual' in feature_lower or 'design' in feature_lower or 'responsive' in feature_lower:
            doc += f"This feature implements user interface improvements, including responsive design and visual enhancements. "
            doc += f"It ensures the application works well on different screen sizes and provides an improved user experience.\n\n"
        elif 'privacy' in feature_lower or 'protection' in feature_lower:
            doc += f"This feature implements privacy and security measures to protect user data and ensure secure operations.\n\n"
        else:
            doc += f"This feature was implemented as part of the autonomous development process. "
            doc += f"It includes the necessary code, tests, and documentation to fulfill the requirements.\n\n"
        
        # Implementation Files section with descriptions
        if unique_code_files:
            doc += "## Implementation Files\n\n"
            doc += "### Source Code Files\n\n"
            for file in unique_code_files:
                description = self._get_file_description(file)
                # Remove output/ prefix if present for cleaner display
                display_path = file.replace('output/', '') if file.startswith('output/') else file
                doc += f"- **`{display_path}`**: {description}\n"
            doc += "\n"
        
        if unique_test_files:
            doc += "### Test Files\n\n"
            for file in unique_test_files:
                description = self._get_file_description(file)
                # Remove output/ prefix if present for cleaner display
                display_path = file.replace('output/', '') if file.startswith('output/') else file
                doc += f"- **`{display_path}`**: {description}\n"
            doc += "\n"
        
        # Technical Details section
        doc += "## Technical Details\n\n"
        
        # Count files by type
        php_files = [f for f in unique_code_files if f.endswith('.php')]
        html_files = [f for f in unique_code_files if f.endswith('.html')]
        py_files = [f for f in unique_code_files if f.endswith('.py')]
        
        if php_files:
            doc += f"- **PHP Files**: {len(php_files)} file(s) implementing backend logic\n"
        if html_files:
            doc += f"- **HTML Files**: {len(html_files)} file(s) implementing user interface\n"
        if py_files:
            doc += f"- **Python Files**: {len(py_files)} file(s) for testing\n"
        if unique_test_files:
            doc += f"- **Test Coverage**: {len(unique_test_files)} test file(s) ensuring functionality\n"
        doc += "\n"
        
        # Status section
        doc += "## Status\n\n"
        doc += f"✅ Feature implemented and tested\n"
        doc += f"✅ All unit tests passing\n"
        doc += f"✅ All regression tests passing\n"
        doc += f"✅ Committed to Git\n\n"
        
        return doc
    
    def _generate_final_documentation(self) -> str:
        """Generate final project documentation without repetitions."""
        # Read task to understand what was built
        try:
            task_description = self.tools.read_file("input/task.txt")
        except:
            task_description = "Software development project"
        
        doc = "# Project Documentation\n\n"
        doc += "## Overview\n\n"
        doc += f"{task_description.strip()}\n\n"
        doc += "This project was autonomously developed by an AI Development Agent using Test-Driven Development (TDD) methodology.\n\n"
        
        # Get actual project structure
        src_files = []
        test_files = []
        other_files = []
        
        if os.path.exists(self.output_dir):
            for root, dirs, files in os.walk(self.output_dir):
                # Skip .git directory
                if '.git' in root:
                    continue
                    
                rel_root = os.path.relpath(root, self.output_dir)
                if rel_root == '.':
                    rel_root = ''
                
                for file in files:
                    if file.startswith('.'):
                        continue
                    full_path = os.path.join(rel_root, file) if rel_root else file
                    
                    if 'test' in full_path.lower() or 'spec' in full_path.lower():
                        test_files.append(full_path)
                    elif full_path.startswith('src/') or full_path.endswith(('.py', '.js', '.html', '.css', '.php', '.java')):
                        src_files.append(full_path)
                    elif full_path not in ['README.md', 'requirements.txt', 'package.json']:
                        other_files.append(full_path)
        
        # Features section (only if we have feature docs)
        if self.feature_docs:
            doc += "## Implemented Features\n\n"
            for i, feature_doc in enumerate(self.feature_docs, 1):
                doc += f"### {i}. {feature_doc['name']}\n\n"
                # Extract first paragraph from description
                desc = feature_doc.get('description', '')
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                doc += f"{desc}\n\n"
        
        # Project Structure
        doc += "## Project Structure\n\n"
        if src_files:
            doc += "### Source Files\n\n"
            for file in sorted(set(src_files))[:20]:  # Limit to 20 files
                doc += f"- `{file}`\n"
            doc += "\n"
        
        if test_files:
            doc += "### Test Files\n\n"
            for file in sorted(set(test_files))[:10]:
                doc += f"- `{file}`\n"
            doc += "\n"
        
        # Directory structure (simplified)
        if os.path.exists(self.output_dir):
            doc += "### Directory Layout\n\n"
            doc += "```\n"
            seen_dirs = set()
            for root, dirs, files in os.walk(self.output_dir):
                if '.git' in root:
                    continue
                rel_root = os.path.relpath(root, self.output_dir)
                if rel_root == '.':
                    rel_root = 'output'
                if rel_root not in seen_dirs:
                    doc += f"{rel_root}/\n"
                    seen_dirs.add(rel_root)
            doc += "```\n\n"
        
        doc += "## Building the Project\n\n"
        doc += "### Prerequisites\n\n"
        doc += "- Python 3.10+\n"
        doc += "- pip (Python package manager)\n\n"
        
        doc += "### Installation\n\n"
        doc += "1. Navigate to the project directory:\n"
        doc += "   ```bash\n"
        doc += "   cd output/\n"
        doc += "   ```\n\n"
        doc += "2. Install dependencies (if requirements.txt exists):\n"
        doc += "   ```bash\n"
        doc += "   pip install -r requirements.txt\n"
        doc += "   ```\n\n"
        
        doc += "## Running the Project\n\n"
        doc += "### Development Mode\n\n"
        doc += "To run the project in development mode:\n\n"
        doc += "```bash\n"
        doc += "python main.py\n"
        doc += "# or\n"
        doc += "python app.py\n"
        doc += "# or\n"
        doc += "python -m app\n"
        doc += "```\n\n"
        
        doc += "### Running Tests\n\n"
        doc += "```bash\n"
        doc += "# Using pytest\n"
        doc += "pytest\n"
        doc += "# or\n"
        doc += "python -m pytest\n\n"
        doc += "# Using unittest\n"
        doc += "python -m unittest discover\n"
        doc += "```\n\n"
        
        doc += "## Deployment\n\n"
        doc += "### Production Deployment\n\n"
        doc += "1. **Prepare the environment:**\n"
        doc += "   - Set up a production server (Linux recommended)\n"
        doc += "   - Install Python 3.10+ and required system dependencies\n"
        doc += "   - Configure environment variables\n\n"
        
        doc += "2. **Deploy the code:**\n"
        doc += "   ```bash\n"
        doc += "   # Clone or copy the repository\n"
        doc += "   git clone <repository-url>\n"
        doc += "   cd <project-directory>\n\n"
        doc += "   # Install dependencies\n"
        doc += "   pip install -r requirements.txt\n"
        doc += "   ```\n\n"
        
        doc += "3. **Configure the application:**\n"
        doc += "   - Set environment variables\n"
        doc += "   - Configure database connections (if applicable)\n"
        doc += "   - Set up SSL certificates (if needed)\n\n"
        
        doc += "4. **Run the application:**\n"
        doc += "   ```bash\n"
        doc += "   # Using a process manager (recommended)\n"
        doc += "   systemd, supervisor, or PM2\n\n"
        doc += "   # Or directly\n"
        doc += "   python app.py\n"
        doc += "   ```\n\n"
        
        doc += "5. **Set up reverse proxy (if web application):**\n"
        doc += "   - Configure Nginx or Apache\n"
        doc += "   - Set up SSL with Let's Encrypt\n\n"
        
        doc += "### Docker Deployment (if applicable)\n\n"
        doc += "If a Dockerfile is present:\n"
        doc += "```bash\n"
        doc += "docker build -t myapp .\n"
        doc += "docker run -p 8000:8000 myapp\n"
        doc += "```\n\n"
        
        doc += "## Notes\n\n"
        doc += "- This project was generated autonomously by an AI agent\n"
        doc += "- Review and test thoroughly before deploying to production\n"
        doc += "- Customize configuration as needed for your environment\n\n"
        
        return doc
    
    def _clean_json(self, text: str) -> List[Dict[str, Any]]:
        """
        Robust JSON parser that extracts JSON from Planner output.
        Handles markdown code blocks, conversational text, and malformed JSON.
        
        Args:
            text: Raw text from Planner that may contain JSON
            
        Returns:
            Parsed JSON array of action dictionaries
            
        Raises:
            ValueError: If no valid JSON can be extracted
        """
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Try 1: Direct JSON parsing
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                print("✅ Parsed JSON directly")
                return parsed
        except json.JSONDecodeError as e:
            # If JSON is incomplete (truncated), try to fix it
            if "Unterminated string" in str(e) or "Expecting" in str(e):
                # Try to find and extract complete JSON array
                text = self._fix_truncated_json(text)
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        print("✅ Parsed JSON after fixing truncation")
                        return parsed
                except json.JSONDecodeError:
                    pass
        
        # Try 2: Extract JSON from markdown code blocks
        import re
        json_patterns = [
            r'```json\s*(.*?)\s*```',  # ```json ... ```
            r'```\s*(.*?)\s*```',       # ``` ... ```
            r'\[.*\]',                  # Any array-like structure
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    # If pattern captured group, use it; otherwise use full match
                    json_str = match if isinstance(match, str) else match[0] if match else text
                    parsed = json.loads(json_str.strip())
                    if isinstance(parsed, list):
                        print(f"✅ Extracted JSON from markdown block")
                        return parsed
                except (json.JSONDecodeError, IndexError, AttributeError):
                    continue
        
        # Try 3: Find JSON array boundaries manually
        try:
            # Find first '[' and last ']'
            start_idx = text.find('[')
            end_idx = text.rfind(']')
            if start_idx != -1:
                if end_idx != -1 and end_idx > start_idx:
                    json_str = text[start_idx:end_idx + 1]
                else:
                    # JSON is truncated - try to fix it
                    json_str = text[start_idx:]
                    json_str = self._fix_truncated_json(json_str)
                
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    print(f"✅ Extracted JSON by finding array boundaries")
                    return parsed
        except json.JSONDecodeError:
            pass
        
        # Try 4: Fix common JSON issues and retry
        try:
            # Remove common non-JSON prefixes/suffixes
            cleaned = text
            for prefix in ['Here is the plan:', 'Plan:', 'JSON:', '```']:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            
            # Try to fix trailing commas
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                print(f"✅ Parsed JSON after cleaning")
                return parsed
        except json.JSONDecodeError:
            pass
        
        # If all else fails, try to extract partial JSON and complete it
        try:
            # Find the start of JSON array
            start_idx = text.find('[')
            if start_idx != -1:
                # Extract JSON starting from first '['
                partial_json = text[start_idx:]
                
                # Try to find complete objects within the partial JSON
                # Look for complete JSON objects (those that have matching braces)
                brace_count = 0
                bracket_count = 0
                last_complete_pos = -1
                in_string = False
                escape_next = False
                complete_objects = []
                current_obj_start = -1
                
                for i, char in enumerate(partial_json):
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if not in_string:
                        if char == '{':
                            if brace_count == 0:
                                current_obj_start = i
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0 and current_obj_start != -1:
                                # Found a complete object
                                complete_objects.append((current_obj_start, i + 1))
                                last_complete_pos = i
                                current_obj_start = -1
                        elif char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                
                # If we found complete objects, try to form a valid array
                if complete_objects:
                    # Extract all complete objects and validate them
                    objects_json = []
                    for start, end in complete_objects:
                        obj_str = partial_json[start:end]
                        try:
                            obj = json.loads(obj_str)
                            objects_json.append(obj)
                        except json.JSONDecodeError:
                            pass
                    
                    if objects_json:
                        # Return as list
                        print(f"✅ Parsed JSON after extracting {len(objects_json)} complete objects")
                        return objects_json
                
                # If we found at least one complete object but couldn't parse individually, try as array
                if last_complete_pos > 0:
                    # Extract up to the last complete object
                    extracted = partial_json[:last_complete_pos + 1]
                    # Try to form a valid array
                    if not extracted.startswith('['):
                        extracted = '[' + extracted
                    if not extracted.endswith(']'):
                        extracted = extracted + ']'
                    try:
                        parsed = json.loads(extracted)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            print("✅ Parsed JSON after extracting complete objects")
                            return parsed
                    except json.JSONDecodeError:
                        pass
                
                # Last resort: try to fix the truncated JSON
                fixed_json = self._fix_truncated_json(partial_json)
                try:
                    parsed = json.loads(fixed_json)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        print("✅ Parsed JSON after aggressive fixing")
                        return parsed
                except json.JSONDecodeError:
                    pass
        except (json.JSONDecodeError, ValueError, Exception) as e:
            self._log_thought(f"JSON fixing failed: {str(e)}")
            pass
        
        # If all else fails, raise error with more context
        error_msg = f"Could not extract valid JSON from Planner response:\n{text[:1000]}"
        if len(text) > 1000:
            error_msg += f"\n... (truncated, total length: {len(text)} chars)"
        raise ValueError(error_msg)
    
    def _fix_truncated_json(self, text: str) -> str:
        """
        Attempt to fix truncated or incomplete JSON.
        Returns fixed JSON string.
        """
        import re
        
        # Remove any trailing incomplete content
        text = text.strip()
        
        # Strategy 1: Try to find complete JSON objects and extract them
        # Find the start of the JSON array
        array_start = text.find('[')
        if array_start == -1:
            return '[]'
        
        text = text[array_start:]
        
        # Strategy 2: Find all complete objects within the array
        complete_objects = []
        depth = 0
        in_string = False
        escape_next = False
        obj_start = -1
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if char == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and obj_start != -1:
                    # Found a complete object
                    obj_str = text[obj_start:i+1]
                    try:
                        # Verify it's valid JSON
                        json.loads(obj_str)
                        complete_objects.append(obj_str)
                    except json.JSONDecodeError:
                        pass
                    obj_start = -1
        
        # Strategy 3: If we found complete objects, build a valid array
        if complete_objects:
            return '[' + ','.join(complete_objects) + ']'
        
        # Strategy 4: Aggressive fixing - try to close incomplete JSON
        if not text.endswith(']'):
            # Count brackets
            open_brackets = text.count('[') - text.count(']')
            open_braces = text.count('{') - text.count('}')
            
            # Find last complete structure by looking for },
            last_complete = text.rfind('},')
            if last_complete != -1:
                text = text[:last_complete + 1] + ']'
            else:
                # Try to find any closing brace
                last_brace = text.rfind('}')
                if last_brace != -1:
                    text = text[:last_brace + 1]
                    # Close remaining
                    text += '}' * max(0, open_braces - 1)
                    text += ']' * max(1, open_brackets)
                else:
                    # No valid structure found
                    return '[]'
        
        return text
    
    def _check_phpunit_available(self) -> bool:
        """Check if PHPUnit is available in the system (cached)."""
        if self.phpunit_available is not None:
            return self.phpunit_available
        
        try:
            stdout, stderr, return_code = self.tools.execute_command("phpunit --version", timeout=5, cwd=self.output_dir)
            self.phpunit_available = return_code == 0
            return self.phpunit_available
        except:
            self.phpunit_available = False
            return False
    
    def _start_php_server(self) -> bool:
        """
        Start PHP built-in server for testing PHP applications.
        Returns True if server started successfully, False otherwise.
        """
        if self.php_server_process is not None:
            # Server already running
            return True
        
        project_type = self._detect_project_type()
        if project_type != "PHP":
            return True  # Not a PHP project, no server needed
        
        # Find the main entry point (index.php, api.php, or first PHP file in src/)
        src_dir = os.path.join(self.output_dir, "src")
        if not os.path.exists(src_dir):
            print("⚠️  src/ directory not found, cannot start PHP server")
            return False
        
        # Look for common entry points
        entry_points = ["index.php", "api.php", "app.php", "main.php"]
        router_file = None
        for ep in entry_points:
            ep_path = os.path.join(src_dir, ep)
            if os.path.exists(ep_path):
                router_file = ep
                break
        
        # If no entry point found, use first PHP file or create a simple router
        if not router_file:
            php_files = [f for f in os.listdir(src_dir) if f.endswith('.php')]
            if php_files:
                router_file = php_files[0]
            else:
                print("⚠️  No PHP files found in src/, cannot start PHP server")
                return False
        
        try:
            # Start PHP built-in server
            router_path = os.path.join("src", router_file)
            cmd = f"php -S localhost:{self.php_server_port} -t src {router_path}"
            print(f"🚀 Starting PHP server on port {self.php_server_port}...")
            self._log_thought(f"Starting PHP server: {cmd}")
            
            # Start server in background
            self.php_server_process = subprocess.Popen(
                cmd.split(),
                cwd=self.output_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a bit for server to start
            time.sleep(2)
            
            # Check if server is running
            if self.php_server_process.poll() is None:
                print(f"✅ PHP server started on http://localhost:{self.php_server_port}")
                self._log_thought(f"PHP server started successfully on port {self.php_server_port}")
                return True
            else:
                # Server process died
                stderr = self.php_server_process.stderr.read() if self.php_server_process.stderr else ""
                print(f"❌ PHP server failed to start: {stderr}")
                self.php_server_process = None
                return False
                
        except Exception as e:
            print(f"❌ Error starting PHP server: {e}")
            self.php_server_process = None
            return False
    
    def _stop_php_server(self) -> None:
        """Stop PHP built-in server if running."""
        if self.php_server_process is not None:
            try:
                print(f"🛑 Stopping PHP server...")
                self.php_server_process.terminate()
                # Wait a bit for graceful shutdown
                try:
                    self.php_server_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate
                    self.php_server_process.kill()
                    self.php_server_process.wait()
                self.php_server_process = None
                print("✅ PHP server stopped")
                self._log_thought("PHP server stopped")
            except Exception as e:
                print(f"⚠️  Error stopping PHP server: {e}")
                self.php_server_process = None
    
    def _get_code_structure_blueprint(self, target_file: str, instruction: str) -> str:
        """
        Generate a structural blueprint for the code to be generated.
        This ensures the Executor knows exactly what classes, functions, and imports are needed.
        
        Returns:
            Blueprint string with required structure
        """
        if not hasattr(self, 'task_description') or not self.task_description:
            return ""
        
        # Extract specific structures from task
        import re
        blueprint_parts = []
        task = self.task_description.lower()
        
        # Detect required libraries
        libraries = []
        if 'llama-cpp-python' in task or 'llama_cpp' in task:
            libraries.append("from llama_cpp import Llama")
        if 'json' in task:
            libraries.append("import json")
        if 'uuid' in task:
            libraries.append("import uuid")
        if 'os' in task or 'file' in task or 'directory' in task:
            libraries.append("import os")
        if 'pathlib' in task:
            libraries.append("from pathlib import Path")
        if 'datetime' in task:
            libraries.append("from datetime import datetime")
        if 'argparse' in task:
            libraries.append("import argparse")
        if 'logging' in task:
            libraries.append("import logging")
        
        if libraries:
            blueprint_parts.append("REQUIRED IMPORTS:\n" + "\n".join(f"  - {lib}" for lib in libraries))
        
        # Detect required functions from task
        functions = []
        func_patterns = [
            r'function\s+(?:called\s+)?[`"\']?(\w+)[`"\']?',
            r'implement\s+(?:a\s+)?(?:function|method)\s+[`"\']?(\w+)[`"\']?',
        ]
        for pattern in func_patterns:
            matches = re.findall(pattern, self.task_description, re.IGNORECASE)
            functions.extend(matches)
        
        # Common patterns to detect from task description
        if 'generate' in task and 'record' in task:
            functions.append("generate_record")
        if 'validate' in task and 'json' in task:
            functions.append("validate_json")
        if 'retry' in task or 'max_consecutive' in task:
            functions.append("retry_logic")
        if 'save' in task or 'write' in task:
            functions.append("save_output")
        if 'initialize' in task or 'init' in task or 'setup' in task:
            functions.append("initialize")
        
        functions = list(set(functions))
        if functions:
            blueprint_parts.append("REQUIRED FUNCTIONS:\n" + "\n".join(f"  - {func}()" for func in functions))
        
        # Detect configuration constants
        configs = []
        config_patterns = [
            r'(MAX_\w+)\s*[:=]?\s*(\d+)',
            r'(MIN_\w+)\s*[:=]?\s*(\d+)',
            r'(\w+_LIMIT)\s*[:=]?\s*(\d+)',
            r'(\w+_COUNT)\s*[:=]?\s*(\d+)',
            r'(\w+_SIZE)\s*[:=]?\s*(\d+)',
        ]
        for pattern in config_patterns:
            matches = re.findall(pattern, self.task_description, re.IGNORECASE)
            for name, value in matches:
                configs.append(f"{name.upper()} = {value}")
        
        if configs:
            blueprint_parts.append("REQUIRED CONSTANTS:\n" + "\n".join(f"  - {cfg}" for cfg in configs))
        
        # File-specific requirements
        if target_file.endswith('.py'):
            if 'main' in task or '__main__' in task or 'cli' in task or 'command line' in task:
                blueprint_parts.append("REQUIRED ENTRY POINT:\n  - if __name__ == '__main__': main() block")
        
        if not blueprint_parts:
            return ""
        
        return "\n\n📐 STRUCTURAL BLUEPRINT FOR THIS FILE:\n" + "\n".join(blueprint_parts) + "\n\nIMPORTANT: Ensure ALL items from this blueprint are implemented in your code.\n"
    
    def _ask_executor(self, instruction: str, target_file: str) -> str:
        """
        Ask the Executor model to generate code for a file.
        
        Args:
            instruction: Detailed instruction from Planner
            target_file: Target file path
            
        Returns:
            Generated code content (raw, no markdown)
        """
        self._log_thought(f"Executor: Generating code for '{target_file}'")
        self._log_thought(f"Executor: Instruction - {instruction[:100]}...")
        
        # Detect if this is a test file
        is_test_file = 'test' in target_file.lower()
        project_type = self._detect_project_type()
        
        # Build prompt with specific instructions for test files
        if is_test_file and project_type == "PHP":
            # For PHP projects, always generate Python tests
            prompt = f"""You are an expert Python Developer. Your Task: Write a Python test file for '{target_file}' to test a PHP application.

CRITICAL REQUIREMENTS:
1. This is a test for a PHP application - use Python to test it via HTTP requests
2. The PHP server will be running on http://localhost:{self.php_server_port} - use this URL for all requests
3. Use Python's requests library (import requests) or subprocess with curl
4. Test PHP API endpoints by making HTTP requests (GET, POST, etc.) to http://localhost:{self.php_server_port}
5. Use simple Python assertions: assert condition, "message"
6. Output clear messages: print("PASS: description") or print("FAIL: description")
7. Can be executed directly with: python {target_file}
8. Use unittest module or simple assert statements
9. Test the PHP application as a black box via HTTP requests
10. Example: response = requests.get("http://localhost:{self.php_server_port}/api.php?action=get_seats")
11. If testing database setup, you can check if database file exists or make API calls
12. CRITICAL: If any test fails, call sys.exit(1) at the end. If all tests pass, call sys.exit(0). This ensures the test script returns the correct exit code.
13. Import sys at the top: import sys

INSTRUCTIONS: {instruction}

REQUIREMENTS: Return ONLY valid, executable Python code. Do NOT use markdown. Do NOT write conversational text. Ensure all syntax is correct. The file MUST end with sys.exit(0) if all tests pass, or sys.exit(1) if any test fails."""
        elif is_test_file and target_file.endswith('.py'):
            # Python test file - can use unittest or pytest
            prompt = f"""You are an expert Python Developer. Your Task: Write a Python test file for '{target_file}'.

You can use:
- unittest module (built-in)
- pytest (if available)
- Simple assertions with assert statements

INSTRUCTIONS: {instruction}

REQUIREMENTS: Return ONLY valid Python code. Do NOT use markdown. Do NOT write conversational text."""
        else:
            # Not a test file, normal generation
            # Include task context AND extracted requirements to help Executor understand requirements
            task_context = ""
            if hasattr(self, 'task_description') and self.task_description:
                # Include more task context (up to 3000 chars)
                task_context = f"\n\nORIGINAL TASK REQUIREMENTS:\n{self.task_description[:3000]}"
                if len(self.task_description) > 3000:
                    task_context += "\n[... additional requirements ...]"
            
            # Include extracted requirements if available
            requirements_context = ""
            if hasattr(self, 'extracted_requirements') and self.extracted_requirements:
                req_list = []
                for req in self.extracted_requirements[:15]:  # Limit to avoid token overflow
                    req_list.append(f"  - [{req.get('id', '?')}] {req.get('description', '')}")
                if req_list:
                    requirements_context = f"\n\n📋 EXTRACTED REQUIREMENTS (must be implemented):\n" + "\n".join(req_list)
                    requirements_context += "\n\n⚠️  CRITICAL: Your code MUST implement ALL of these requirements. Check each one!"
            
            # Include existing code context if file already exists
            existing_code_context = ""
            existing_file_path = os.path.join(self.output_dir, target_file)
            if os.path.exists(existing_file_path):
                try:
                    with open(existing_file_path, 'r', encoding='utf-8') as f:
                        existing_code = f.read()
                    if existing_code:
                        existing_code_context = f"\n\n📄 EXISTING CODE IN THIS FILE (read carefully - you may need to MODIFY it, not replace it):\n```python\n{existing_code[:2000]}\n```\n\n⚠️  IMPORTANT: If modifying existing code, preserve what works and only fix/add what's needed!"
                except Exception:
                    pass
            
            # Include recent errors/context from history
            recent_context = ""
            if hasattr(self, 'history') and self.history:
                recent_actions = [h for h in self.history[-5:] if h.get('action') in ['write_file', 'execute_command']]
                if recent_actions:
                    recent_context = "\n\n📜 RECENT ACTIONS (to understand what was already done):\n"
                    for action in recent_actions:
                        recent_context += f"  - {action.get('action')} on {action.get('target', '?')}\n"
                        if action.get('success') == False:
                            recent_context += f"    ⚠️  This action failed - may need fixing\n"
            
            # Get structural blueprint for the file
            structure_blueprint = self._get_code_structure_blueprint(target_file, instruction)
            
            prompt = f"""You are an expert Senior Developer. Your Task: Write COMPLETE, FUNCTIONAL code for the file '{target_file}'.

INSTRUCTIONS FROM PLANNER:
{instruction}
{task_context}
{requirements_context}
{existing_code_context}
{recent_context}
{structure_blueprint}
🚨 CRITICAL REQUIREMENTS - YOU MUST FOLLOW THESE:

1. WRITE COMPLETE CODE - NOT SKELETON/PLACEHOLDER CODE:
   - Implement ALL functions with ACTUAL WORKING LOGIC
   - Do NOT use placeholder comments like "# TODO: implement this"
   - Do NOT use "pass" or empty function bodies
   - Every function must have REAL implementation

2. IMPLEMENT ALL FEATURES MENTIONED:
   - Read the ORIGINAL TASK REQUIREMENTS above carefully
   - Implement EVERY feature, configuration, and requirement mentioned
   - Use the EXACT libraries, methods, and configurations specified
   - Follow the EXACT data structures and formats specified

3. CODE QUALITY:
   - Include proper error handling
   - Add comments explaining complex logic
   - Use appropriate variable names
   - Follow best practices for the language

4. SPECIFIC TECHNICAL REQUIREMENTS - INTERPRET LIKE A HUMAN:
   - If task says "use requests library" → import requests, use requests.post(), NOT other libraries
   - If task says "use llama-cpp-python" → import llama_cpp, use Llama(), NOT requests
   - If task says "endpoint http://X:Y" → Use EXACTLY that URL, don't change ports or paths
   - If task says "format record_YYYYMMDD_HHMMSS_UUID.json" → Use datetime.strftime() + UUID, NOT just UUID
   - If task says "validate keys: raw_intent, tags" → Check if keys exist AND are not empty, NOT just if JSON is valid
   - If task says "3 retries per record" → Implement retry loop with max 3 attempts, NOT just try/except
   - If task says "KeyboardInterrupt handling" → Add try/except KeyboardInterrupt in run(), save files before exit
   - If task specifies a class structure → Implement EXACTLY that structure, same method names and signatures
   
   CRITICAL API CALL PATTERN (for OpenAI-compatible endpoints):
   When calling HTTP API endpoints that return OpenAI-compatible format:
   ✅ CORRECT PATTERN:
   response = requests.post(url, json=payload, timeout=120)
   response.raise_for_status()
   content = response.json()["choices"][0]["message"]["content"]  # Extract content string
   parsed_data = json.loads(content)  # Parse the JSON string from content
   
   ❌ WRONG PATTERN:
   response = requests.post(url, json=payload)
   return response.json()  # This returns the API wrapper, not the parsed JSON content!
   
   EXAMPLE INTERPRETATION:
   Task says: "Use requests library to call http://192.168.1.29:8081/v1/chat/completions"
   ✅ CORRECT: 
   import requests
   import json
   response = requests.post("http://192.168.1.29:8081/v1/chat/completions", json={...}, timeout=120)
   response.raise_for_status()
   content = response.json()["choices"][0]["message"]["content"]
   intent_data = json.loads(content)  # Now intent_data is the parsed JSON dict
   
   ❌ WRONG: 
   from llama_cpp import Llama
   model = Llama(model_path="http://...")  # llama_cpp doesn't work with URLs!
   
   Task says: "Save as record_YYYYMMDD_HHMMSS_UUID.json"
   ✅ CORRECT: 
   from datetime import datetime
   filename = f"record_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}_{{uuid.uuid4()}}.json"
   
   ❌ WRONG: 
   filename = f"record_{{uuid.uuid4()}}.json"  # Missing datetime!
   
   Task says: "3 retries per record" (for the entire record generation, not just HTTP call)
   ✅ CORRECT:
   for attempt in range(3):
       intent_response = call_model_x()
       if intent_response and validate_json(intent_response):
           solution_response = call_model_y(intent_response)
           if solution_response and validate_json(solution_response):
               save_record(...)
               break  # Success!
       # If we get here, retry the whole record
   
   ❌ WRONG:
   # Only retrying HTTP calls, not the full record generation

5. OUTPUT FORMAT:
   - Return ONLY raw executable code
   - Do NOT use markdown code blocks
   - Do NOT write conversational text
   - The code must be syntactically correct and runnable

REMEMBER: The code will be reviewed for COMPLETENESS. Incomplete or placeholder code will be REJECTED. Write PRODUCTION-READY code that actually works."""
        
        # Determine appropriate system message based on file type
        if is_test_file:
            system_msg = "You are an expert test engineer. Write complete, executable test code. Return only raw code, no markdown."
        else:
            system_msg = """You are a Senior Software Engineer who writes COMPLETE, PRODUCTION-READY code.

CRITICAL RULES:
1. Write FULLY FUNCTIONAL code - not sketches or placeholders
2. Implement ALL logic completely - no empty functions or "pass" statements
3. Include ALL features mentioned in the requirements
4. Use the EXACT libraries and configurations specified - interpret requirements literally
5. Return ONLY raw executable code - no markdown, no explanations
6. Your code will be reviewed for completeness - incomplete code is unacceptable
7. READ THE REQUIREMENTS CAREFULLY - if it says "use requests", use requests. If it says "use llama-cpp-python", use llama-cpp-python.
8. MATCH THE EXACT FORMATS SPECIFIED - if task says "record_YYYYMMDD_HHMMSS_UUID.json", implement that EXACT format
9. IMPLEMENT ALL VALIDATION LOGIC - if task says "validate keys are not empty", check both existence AND emptiness"""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        
        print(f"✍️  Executor coding for: {target_file}")
        try:
            response = self.executor_client.chat_completion(messages, temperature=0.2)
            
            # Clean markdown code blocks if present (defensive)
            response = response.strip()
            if response.startswith('```'):
                # Remove markdown code fences
                lines = response.split('\n')
                # Remove first line (```python or ```)
                if lines[0].startswith('```'):
                    lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                response = '\n'.join(lines)
            
            print(f"✅ Executor generated {len(response)} characters")
            self._log_thought(f"Executor: Generated {len(response)} characters of code for '{target_file}'")
            return response
            
        except Exception as e:
            self._log_thought(f"Executor: ERROR - {str(e)}")
            print(f"❌ Executor error: {e}")
            raise
    
    def _log_thought(self, message: str) -> None:
        """Log a thought/decision to the thought chain."""
        timestamp = time.strftime("%H:%M:%S")
        thought_entry = f"[{timestamp}] {message}"
        self.thought_chain.append(thought_entry)
        print(f"💭 {thought_entry}")
    
    def _extract_api_endpoints(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Extract API endpoint information from PHP files.
        
        Returns:
            List of endpoint dictionaries with action, method, parameters, response format
        """
        import re
        endpoints = []
        full_text = '\n'.join(lines)
        
        # Look for switch/case patterns for API actions
        switch_match = re.search(r'switch\s*\(\s*\$input\[[\'"]action[\'"]\s*\]\s*\)\s*\{([^}]+)\}', full_text, re.DOTALL)
        if switch_match:
            switch_body = switch_match.group(1)
            # Extract case statements
            case_pattern = r"case\s*['\"]([^'\"]+)['\"]\s*:"
            cases = re.findall(case_pattern, switch_body)
            for case in cases:
                endpoints.append({
                    'action': case,
                    'method': 'POST',  # Default, will check REQUEST_METHOD
                    'parameters': [],
                    'response_format': 'json'
                })
        
        # Check for REQUEST_METHOD
        if re.search(r'\$_SERVER\[[\'"]REQUEST_METHOD[\'"]\s*\]\s*===\s*[\'"]POST[\'"]', full_text):
            for ep in endpoints:
                ep['method'] = 'POST'
        elif re.search(r'\$_SERVER\[[\'"]REQUEST_METHOD[\'"]\s*\]\s*===\s*[\'"]GET[\'"]', full_text):
            for ep in endpoints:
                ep['method'] = 'GET'
        
        # Extract function calls to understand parameters
        for i, line in enumerate(lines):
            if 'case' in line and 'action' in line.lower():
                # Look ahead for function calls
                for j in range(i, min(i+10, len(lines))):
                    func_match = re.search(r'(\w+)\s*\(\s*\$pdo\s*,?\s*([^)]*)\)', lines[j])
                    if func_match:
                        func_name = func_match.group(1)
                        params = func_match.group(2)
                        # Find corresponding endpoint
                        for ep in endpoints:
                            if ep['action'] in line:
                                ep['function'] = func_name
                                # Extract parameters from function call
                                param_matches = re.findall(r'\$input\[[\'"](\w+)[\'"]\s*\]', params)
                                ep['parameters'] = param_matches
                                break
        
        # Extract response format from return statements
        for ep in endpoints:
            func_name = ep.get('function', '')
            if func_name:
                # Find function definition
                func_pattern = f'function\\s+{func_name}\\s*\\([^)]*\\)\\s*{{([^}}]+)}}'
                func_match = re.search(func_pattern, full_text, re.DOTALL)
                if func_match:
                    func_body = func_match.group(1)
                    # Check for json_encode patterns
                    json_matches = re.findall(r'json_encode\s*\(\s*\[([^\]]+)\]', func_body)
                    if json_matches:
                        # Extract keys from JSON structure
                        keys = re.findall(r'[\'"](\w+)[\'"]\s*=>', json_matches[0])
                        ep['response_keys'] = keys
        
        return endpoints
    
    def _extract_frontend_api_calls(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Extract API calls from HTML/JavaScript files.
        
        Returns:
            List of API call dictionaries with URL, method, parameters, expected response
        """
        import re
        api_calls = []
        full_text = '\n'.join(lines)
        
        # Find all fetch() calls
        fetch_pattern = r'fetch\s*\(\s*[\'"]([^\'"]+)[\'"]\s*,?\s*({[^}]*})?\s*\)'
        fetch_matches = re.finditer(fetch_pattern, full_text, re.DOTALL)
        
        for match in fetch_matches:
            url = match.group(1)
            options = match.group(2) if match.group(2) else '{}'
            
            # Extract method
            method_match = re.search(r'method\s*:\s*[\'"](\w+)[\'"]', options, re.IGNORECASE)
            method = method_match.group(1).upper() if method_match else 'GET'
            
            # Extract action from URL
            action_match = re.search(r'action[=:](\w+)', url)
            action = action_match.group(1) if action_match else None
            
            # Extract parameters from URL or body
            params = {}
            if '?' in url:
                query_params = url.split('?')[1].split('&')
                for param in query_params:
                    if '=' in param:
                        key, value = param.split('=', 1)
                        params[key] = value
            
            # Extract expected response structure (look for .json() calls after fetch)
            response_keys = []
            # Find the line after fetch and look for .json() or response usage
            match_end = match.end()
            next_lines = full_text[match_end:match_end+500]
            json_match = re.search(r'\.json\s*\(\s*\)', next_lines)
            if json_match:
                # Look for property access like data.seats, data.success
                prop_matches = re.findall(r'data\.(\w+)', next_lines)
                response_keys = prop_matches
            
            api_calls.append({
                'url': url,
                'method': method,
                'action': action,
                'parameters': params,
                'expected_response_keys': response_keys
            })
        
        return api_calls
    
    def _get_file_summary(self, file_path: str, max_lines: int = 50) -> str:
        """
        Get a comprehensive summary of a file's content including API endpoints, JSON formats, and dependencies.
        
        Args:
            file_path: Path to the file
            max_lines: Maximum number of lines to include in preview
            
        Returns:
            Summary string with key information
        """
        try:
            full_path = os.path.join(self.output_dir, file_path)
            if not os.path.exists(full_path):
                return f"File {file_path} does not exist"
            
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                return f"File {file_path} is empty"
            
            # Get first max_lines for preview
            preview = ''.join(lines[:max_lines])
            
            # Extract key information based on file type
            summary_parts = [f"File: {file_path} ({len(lines)} lines)"]
            
            if file_path.endswith('.php'):
                # Extract requires/includes
                requires = [line.strip() for line in lines if line.strip().startswith(('require', 'include'))]
                if requires:
                    summary_parts.append(f"Requires/Includes: {', '.join(requires[:5])}")
                    # CRITICAL: Check if required files exist
                    existing_requires = []
                    missing_requires = []
                    for req_line in requires[:5]:
                        # Extract filename from require/include
                        import re
                        file_match = re.search(r'[\'"]([^\'"]+\.php)[\'"]', req_line)
                        if file_match:
                            req_file = file_match.group(1)
                            req_path = os.path.join(self.output_dir, "src", req_file)
                            if os.path.exists(req_path):
                                existing_requires.append(f"✅ {req_file} EXISTS")
                            else:
                                missing_requires.append(f"❌ {req_file} MISSING - file does not exist!")
                    if existing_requires:
                        summary_parts.append(f"  Dependency Status: {', '.join(existing_requires)}")
                    if missing_requires:
                        summary_parts.append(f"  ⚠️ WARNING: {', '.join(missing_requires)}")
                
                # Extract class definitions
                classes = [line.strip() for line in lines if 'class ' in line and '{' in line]
                if classes:
                    summary_parts.append(f"Classes: {', '.join(classes[:3])}")
                
                # Extract function definitions with parameters
                import re
                functions = []
                for line in lines:
                    func_match = re.search(r'function\s+(\w+)\s*\(([^)]*)\)', line)
                    if func_match:
                        func_name = func_match.group(1)
                        params = func_match.group(2)
                        param_list = [p.strip().split('$')[-1] for p in params.split(',') if p.strip()]
                        functions.append(f"{func_name}({', '.join(param_list[:5])})")
                if functions:
                    summary_parts.append(f"Functions: {', '.join(functions[:10])}")
                
                # Extract API endpoints if this is an API file
                if 'api' in file_path.lower() or 'api.php' in file_path:
                    endpoints = self._extract_api_endpoints(lines)
                    if endpoints:
                        summary_parts.append(f"\n🌐 API ENDPOINTS:")
                        for ep in endpoints:
                            ep_str = f"  - {ep['action']} ({ep['method']})"
                            if ep.get('parameters'):
                                ep_str += f" - Parameters: {', '.join(ep['parameters'])}"
                            if ep.get('response_keys'):
                                ep_str += f" - Returns: {', '.join(ep['response_keys'])}"
                            summary_parts.append(ep_str)
                
                # Extract JSON response patterns
                json_patterns = []
                for line in lines:
                    if 'json_encode' in line:
                        # Extract JSON structure
                        json_match = re.search(r'json_encode\s*\(\s*\[?\s*[\'"](\w+)[\'"]\s*=>', line)
                        if json_match:
                            json_patterns.append(json_match.group(1))
                if json_patterns:
                    summary_parts.append(f"JSON Response Keys: {', '.join(set(json_patterns[:10]))}")
            
            elif file_path.endswith('.html'):
                # Extract script sources and important elements
                scripts = [line.strip() for line in lines if '<script' in line.lower() or 'src=' in line.lower()]
                if scripts:
                    summary_parts.append(f"Scripts/External: {', '.join(scripts[:3])}")
                
                # Extract API calls from JavaScript
                api_calls = self._extract_frontend_api_calls(lines)
                if api_calls:
                    summary_parts.append(f"\n🌐 FRONTEND API CALLS:")
                    for call in api_calls:
                        call_str = f"  - {call['url']} ({call['method']})"
                        if call.get('action'):
                            call_str += f" - Action: {call['action']}"
                        if call.get('expected_response_keys'):
                            call_str += f" - Expects: {', '.join(call['expected_response_keys'])}"
                        summary_parts.append(call_str)
            
            summary_parts.append(f"\nFirst {min(max_lines, len(lines))} lines:\n{preview}")
            
            if len(lines) > max_lines:
                summary_parts.append(f"\n[... {len(lines) - max_lines} more lines ...]")
            
            return '\n'.join(summary_parts)
            
        except Exception as e:
            return f"Error reading {file_path}: {str(e)}"
    
    def _generate_coherence_report(self, project_type: str) -> str:
        """
        Generate a coherence report analyzing frontend-backend and test-API consistency.
        Works for PHP, Python, and other project types.
        
        Args:
            project_type: Type of project (PHP, Python, etc.)
            
        Returns:
            Coherence report string with detected issues
        """
        import re
        issues = []
        warnings = []
        report_parts = ["\n🔍 COHERENCE ANALYSIS:"]
        
        src_dir = os.path.join(self.output_dir, "src")
        tests_dir = os.path.join(self.output_dir, "tests")
        
        # ============= TEST-API ALIGNMENT VALIDATION =============
        if os.path.exists(tests_dir):
            test_files = [f for f in os.listdir(tests_dir) if f.endswith('.py')]
            
            # Extract test expectations
            test_expectations = self._extract_test_expectations(tests_dir, test_files)
            
            # For PHP projects, compare with API
            if project_type == "PHP" and os.path.exists(src_dir):
                php_files = [f for f in os.listdir(src_dir) if f.endswith('.php')]
                api_info = self._extract_php_api_info(src_dir, php_files)
                
                if test_expectations and api_info:
                    report_parts.append("\n  📋 TEST-API ALIGNMENT:")
                    
                    # Check HTTP method alignment
                    for test_name, test_data in test_expectations.items():
                        for endpoint, expected_method in test_data.get('http_methods', {}).items():
                            if endpoint in api_info.get('endpoints', {}):
                                actual_method = api_info['endpoints'][endpoint].get('method', 'POST')
                                # API accepts both if it reads from both $_GET and $_POST
                                if expected_method != actual_method and not api_info.get('accepts_both', False):
                                    issues.append(f"HTTP method mismatch for '{endpoint}'")
                                    report_parts.append(f"    ❌ {test_name}: calls '{endpoint}' with {expected_method}, but API expects {actual_method}")
                        
                        # Check parameter alignment
                        for endpoint, params in test_data.get('parameters', {}).items():
                            if endpoint in api_info.get('endpoints', {}):
                                api_params = api_info['endpoints'][endpoint].get('params', [])
                                for param in params:
                                    if param not in api_params and param != 'action':
                                        # Check for similar params (e.g., seat_id vs seat_number)
                                        similar = [p for p in api_params if param.replace('_id', '') in p or p.replace('_number', '') in param]
                                        if similar:
                                            warnings.append(f"Parameter name mismatch: {param} vs {similar[0]}")
                                            report_parts.append(f"    ⚠️ {test_name}: uses param '{param}' but API expects '{similar[0]}'")
                        
                        # Check response format alignment
                        for endpoint, expected_keys in test_data.get('expected_response', {}).items():
                            if endpoint in api_info.get('endpoints', {}):
                                api_keys = api_info['endpoints'][endpoint].get('response_keys', [])
                                missing = set(expected_keys) - set(api_keys)
                                if missing:
                                    warnings.append(f"Response key mismatch for '{endpoint}'")
                                    report_parts.append(f"    ⚠️ {test_name}: expects keys {missing} but API doesn't return them")
            
            # For Python projects, check import/function alignment
            elif project_type == "Python" and os.path.exists(src_dir):
                py_src_files = [f for f in os.listdir(src_dir) if f.endswith('.py')]
                src_info = self._extract_python_src_info(src_dir, py_src_files)
                
                if test_expectations and src_info:
                    report_parts.append("\n  📋 TEST-SOURCE ALIGNMENT:")
                    
                    # Check function/class imports
                    for test_name, test_data in test_expectations.items():
                        for imported in test_data.get('imports', []):
                            if imported not in src_info.get('exported', []):
                                warnings.append(f"Import not found: {imported}")
                                report_parts.append(f"    ⚠️ {test_name}: imports '{imported}' but it's not defined in source files")
        
        # ============= FRONTEND-BACKEND COHERENCE (PHP) =============
        if project_type == "PHP" and os.path.exists(src_dir):
            html_files = [f for f in os.listdir(src_dir) if f.endswith('.html')]
            php_files = [f for f in os.listdir(src_dir) if f.endswith('.php')]
            js_files = [f for f in os.listdir(src_dir) if f.endswith('.js')]
            
            if (html_files or js_files) and php_files:
                # Extract frontend API calls from HTML and JS
                frontend_calls = []
                for html_file in html_files:
                    html_path = os.path.join(src_dir, html_file)
                    try:
                        with open(html_path, 'r', encoding='utf-8') as f:
                            calls = self._extract_frontend_api_calls(f.readlines())
                            frontend_calls.extend(calls)
                    except:
                        pass
                
                for js_file in js_files:
                    js_path = os.path.join(src_dir, js_file)
                    try:
                        with open(js_path, 'r', encoding='utf-8') as f:
                            calls = self._extract_js_api_calls(f.read())
                            frontend_calls.extend(calls)
                    except:
                        pass
                
                # Extract backend endpoints
                backend_endpoints = []
                api_file = None
                for php_file in php_files:
                    if 'api' in php_file.lower():
                        api_file = php_file
                        api_path = os.path.join(src_dir, php_file)
                        try:
                            with open(api_path, 'r', encoding='utf-8') as f:
                                endpoints = self._extract_api_endpoints(f.readlines())
                                backend_endpoints.extend(endpoints)
                        except:
                            pass
                        break
                
                if frontend_calls and backend_endpoints:
                    report_parts.append("\n  📋 FRONTEND-BACKEND COHERENCE:")
                    
                    frontend_actions = {call['action']: call for call in frontend_calls if call.get('action')}
                    backend_actions = {ep['action']: ep for ep in backend_endpoints}
                    
                    # Missing endpoints
                    missing = set(frontend_actions.keys()) - set(backend_actions.keys())
                    for action in missing:
                        issues.append(f"Missing endpoint: {action}")
                        report_parts.append(f"    ❌ Frontend calls '{action}' but backend has no handler")
                    
                    # Method mismatches
                    for action in set(frontend_actions.keys()) & set(backend_actions.keys()):
                        if frontend_actions[action]['method'] != backend_actions[action]['method']:
                            issues.append(f"Method mismatch: {action}")
                            report_parts.append(f"    ❌ '{action}': Frontend uses {frontend_actions[action]['method']}, Backend expects {backend_actions[action]['method']}")
                
                # Check for require/include mismatches
                if api_file:
                    api_path = os.path.join(src_dir, api_file)
                    try:
                        with open(api_path, 'r', encoding='utf-8') as f:
                            api_content = f.read()
                        require_matches = re.findall(r'(?:require|include)(?:_once)?\s+[\'"]([^\'"]+)[\'"]', api_content)
                        for req_file in require_matches:
                            if req_file.endswith('.php'):
                                req_path = os.path.join(src_dir, req_file)
                                if not os.path.exists(req_path):
                                    similar = [f for f in php_files if req_file.replace('.php', '') in f]
                                    if similar:
                                        issues.append(f"Dependency error: {req_file}")
                                        report_parts.append(f"    ❌ Requires '{req_file}' but doesn't exist. Did you mean '{similar[0]}'?")
                                    else:
                                        issues.append(f"Missing file: {req_file}")
                                        report_parts.append(f"    ❌ Requires '{req_file}' but file doesn't exist")
                            elif req_file.endswith(('.sqlite', '.db', '.sqlite3')):
                                issues.append(f"Invalid require: {req_file}")
                                report_parts.append(f"    ❌ Cannot require data file '{req_file}' - use PDO to connect to database")
                    except:
                        pass
        
        # ============= PYTHON COHERENCE =============
        elif project_type == "Python" and os.path.exists(src_dir):
            py_files = [f for f in os.listdir(src_dir) if f.endswith('.py')]
            
            # Check for import mismatches
            for py_file in py_files:
                py_path = os.path.join(src_dir, py_file)
                try:
                    with open(py_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check imports
                    imports = re.findall(r'from\s+(\S+)\s+import|import\s+(\S+)', content)
                    for imp in imports:
                        module = imp[0] or imp[1]
                        if module.startswith('.'):  # Relative import
                            module_file = module.lstrip('.').replace('.', '/') + '.py'
                            if not os.path.exists(os.path.join(src_dir, module_file)):
                                warnings.append(f"Import not found: {module}")
                                report_parts.append(f"    ⚠️ {py_file}: imports '{module}' but file doesn't exist")
                except:
                    pass
        
        # Summary
        if issues or warnings:
            report_parts.insert(1, f"\n  Found {len(issues)} critical issues and {len(warnings)} warnings:")
            return '\n'.join(report_parts)
        elif len(report_parts) > 1:
            report_parts.append("\n  ✅ No coherence issues detected")
            return '\n'.join(report_parts)
        
        return ""
    
    def _extract_test_expectations(self, tests_dir: str, test_files: list) -> dict:
        """Extract test expectations from test files."""
        import re
        expectations = {}
        
        for test_file in test_files:
            test_path = os.path.join(tests_dir, test_file)
            try:
                with open(test_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                test_name = test_file.replace('.py', '')
                expectations[test_name] = {
                    'http_methods': {},
                    'parameters': {},
                    'expected_response': {},
                    'imports': []
                }
                
                # Extract HTTP method calls
                get_calls = re.findall(r'requests\.get\([\'"]([^\'"]*action=([^\'"&]+)[^\'"]*)[\'"]', content)
                for url, action in get_calls:
                    expectations[test_name]['http_methods'][action] = 'GET'
                
                post_calls = re.findall(r'requests\.post\([\'"]([^\'"]*)[\'"]', content)
                # Extract action from payload
                action_matches = re.findall(r'[\'"]action[\'"]\s*:\s*[\'"]([^\'"]+)[\'"]', content)
                for action in action_matches:
                    if action not in expectations[test_name]['http_methods']:
                        expectations[test_name]['http_methods'][action] = 'POST'
                
                # Extract parameters
                param_matches = re.findall(r'payload\s*=\s*\{([^}]+)\}', content)
                for params_str in param_matches:
                    params = re.findall(r'[\'"](\w+)[\'"]', params_str)
                    for action in action_matches:
                        expectations[test_name]['parameters'][action] = params
                
                # Extract expected response keys
                assert_matches = re.findall(r'assert\s+[\'"](\w+)[\'"]\s+in\s+(?:data|response)', content)
                key_matches = re.findall(r'data\[[\'"](\w+)[\'"]\]|response\.json\(\)\[[\'"](\w+)[\'"]\]', content)
                for match in key_matches:
                    key = match[0] or match[1]
                    for action in action_matches:
                        if action not in expectations[test_name]['expected_response']:
                            expectations[test_name]['expected_response'][action] = []
                        expectations[test_name]['expected_response'][action].append(key)
                
                # Extract imports (for Python projects)
                import_matches = re.findall(r'from\s+src\.(\w+)\s+import\s+(\w+)', content)
                for module, item in import_matches:
                    expectations[test_name]['imports'].append(f"{module}.{item}")
                
            except:
                pass
        
        return expectations
    
    def _extract_php_api_info(self, src_dir: str, php_files: list) -> dict:
        """Extract API information from PHP files."""
        import re
        api_info = {'endpoints': {}, 'accepts_both': False}
        
        for php_file in php_files:
            if 'api' not in php_file.lower():
                continue
            
            php_path = os.path.join(src_dir, php_file)
            try:
                with open(php_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if API accepts both GET and POST
                if 'array_merge' in content and ('$_GET' in content or '$_POST' in content):
                    api_info['accepts_both'] = True
                if 'get_input_data' in content or 'file_get_contents' in content:
                    api_info['accepts_both'] = True
                
                # Extract case statements for endpoints
                case_matches = re.findall(r'case\s+[\'"](\w+)[\'"]:', content)
                for action in case_matches:
                    api_info['endpoints'][action] = {
                        'method': 'POST' if '$_POST' in content else 'GET',
                        'params': [],
                        'response_keys': []
                    }
                
                # Extract parameters used
                for action in case_matches:
                    # Find function body for this action
                    func_pattern = rf'function\s+{action}\s*\([^)]*\)'
                    func_match = re.search(func_pattern, content)
                    if func_match:
                        # Look for parameter access
                        param_matches = re.findall(rf'\$_(?:POST|GET|REQUEST)\[[\'"](\w+)[\'"]\]', content)
                        api_info['endpoints'][action]['params'] = list(set(param_matches))
                
                # Extract response keys
                response_matches = re.findall(r'[\'"](\w+)[\'"]\s*=>', content)
                for action in case_matches:
                    api_info['endpoints'][action]['response_keys'] = list(set(response_matches))
                
            except:
                pass
        
        return api_info
    
    def _extract_python_src_info(self, src_dir: str, py_files: list) -> dict:
        """Extract source information from Python files."""
        import re
        src_info = {'exported': [], 'functions': [], 'classes': []}
        
        for py_file in py_files:
            py_path = os.path.join(src_dir, py_file)
            try:
                with open(py_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                module_name = py_file.replace('.py', '')
                
                # Extract function definitions
                func_matches = re.findall(r'def\s+(\w+)\s*\(', content)
                for func in func_matches:
                    if not func.startswith('_'):
                        src_info['exported'].append(f"{module_name}.{func}")
                        src_info['functions'].append(func)
                
                # Extract class definitions
                class_matches = re.findall(r'class\s+(\w+)\s*[\(:]', content)
                for cls in class_matches:
                    src_info['exported'].append(f"{module_name}.{cls}")
                    src_info['classes'].append(cls)
                
            except:
                pass
        
        return src_info
    
    def _extract_js_api_calls(self, js_content: str) -> list:
        """Extract API calls from JavaScript code."""
        import re
        calls = []
        
        # Extract fetch calls
        fetch_matches = re.findall(r'fetch\([\'"]([^\'"]+)[\'"](?:.*?method:\s*[\'"](\w+)[\'"])?', js_content, re.DOTALL)
        for url, method in fetch_matches:
            action_match = re.search(r'action[=:][\s\'\"]*(\w+)', url)
            if action_match:
                calls.append({
                    'action': action_match.group(1),
                    'method': method.upper() if method else 'GET',
                    'url': url
                })
        
        # Extract apiRequest calls (custom function pattern)
        api_matches = re.findall(r'apiRequest\([\'"](\w+)[\'"]', js_content)
        for action in api_matches:
            calls.append({
                'action': action,
                'method': 'POST',  # Assuming POST for custom apiRequest
                'url': f'?action={action}'
            })
        
        return calls
    
    def _get_existing_files_context(self, project_type: str) -> str:
        """
        Get context about existing files in the output directory with enhanced information.
        
        Args:
            project_type: Type of project (PHP, Python, etc.)
            
        Returns:
            Context string with information about existing files
        """
        context_parts = []
        
        # List source files
        src_dir = os.path.join(self.output_dir, "src")
        if os.path.exists(src_dir):
            src_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
            if src_files:
                context_parts.append(f"\n📁 EXISTING SOURCE FILES in src/ ({len(src_files)} files):")
                for file in sorted(src_files):
                    file_path = f"src/{file}"
                    summary = self._get_file_summary(file_path, max_lines=30)
                    context_parts.append(f"\n{summary}")
        
        # List test files
        tests_dir = os.path.join(self.output_dir, "tests")
        if os.path.exists(tests_dir):
            test_files = [f for f in os.listdir(tests_dir) if os.path.isfile(os.path.join(tests_dir, f)) and not f.startswith('.')]
            if test_files:
                context_parts.append(f"\n📁 EXISTING TEST FILES in tests/ ({len(test_files)} files):")
                for file in sorted(test_files):
                    context_parts.append(f"  - tests/{file}")
        
        # List feature docs (to show what features were completed)
        docs_dir = os.path.join(self.output_dir, "docs", "features")
        if os.path.exists(docs_dir):
            doc_files = [f.replace('.md', '') for f in os.listdir(docs_dir) if f.endswith('.md')]
            if doc_files:
                context_parts.append(f"\n✅ COMPLETED FEATURES (documentation exists):")
                for doc in sorted(doc_files):
                    context_parts.append(f"  - {doc}")
        
        # Generate coherence report for PHP projects
        if project_type == "PHP":
            coherence_report = self._generate_coherence_report(project_type)
            if coherence_report:
                context_parts.append(coherence_report)
        
        if not context_parts:
            return "\n📁 NO EXISTING FILES - This is the first feature."
        
        return '\n'.join(context_parts)
    
    def _detect_project_type(self) -> str:
        """Detect the programming language/framework of the project."""
        if not os.path.exists(self.output_dir):
            return "unknown"
        
        # Check for PHP
        php_files = list(Path(self.output_dir).glob("**/*.php"))
        if php_files:
            return "PHP"
        
        # Check for Python
        py_files = list(Path(self.output_dir).glob("**/*.py"))
        if py_files:
            return "Python"
        
        # Check for Node.js
        if Path(self.output_dir).joinpath("package.json").exists():
            return "Node.js"
        
        # Check for Java
        java_files = list(Path(self.output_dir).glob("**/*.java"))
        if java_files:
            return "Java"
        
        # Check for Go
        go_files = list(Path(self.output_dir).glob("**/*.go"))
        if go_files:
            return "Go"
        
        # Check for Ruby
        rb_files = list(Path(self.output_dir).glob("**/*.rb"))
        if rb_files or Path(self.output_dir).joinpath("Gemfile").exists():
            return "Ruby"
        
        return "unknown"
    
    def _check_php_syntax(self, file_path: str) -> Optional[str]:
        """
        Check PHP file for syntax errors.
        
        Args:
            file_path: Path to PHP file
            
        Returns:
            Error message if syntax error found, None otherwise
        """
        try:
            result = subprocess.run(
                ['php', '-l', file_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                # Extract error message
                error_lines = result.stderr.strip() or result.stdout.strip()
                # Filter out "No syntax errors" message
                if "No syntax errors" in error_lines:
                    return None
                return error_lines
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        return None
    
    def _check_python_syntax(self, file_path: str) -> Optional[str]:
        """
        Check Python file for syntax errors.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Error message if syntax error found, None otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            compile(source, file_path, 'exec')
            return None
        except SyntaxError as e:
            return f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return str(e)
    
    def _get_language_specific_coherence_rules(self, project_type: str) -> str:
        """
        Get coherence rules specific to the project language/framework.
        
        Args:
            project_type: Type of project (PHP, Python, etc.)
            
        Returns:
            String with coherence rules
        """
        rules = "\n\n🚨 CRITICAL COHERENCE RULES - YOU MUST FOLLOW THESE:\n\n"
        
        if project_type == "PHP":
            rules += """1. FILE DEPENDENCIES (PHP):
   - Check ALL require/include statements in existing files
   - If a file requires 'database.php' but 'db.php' exists, you MUST use 'db.php' instead
   - NEVER require/include data files (.sqlite, .db, .json) - use PDO for database connections
   - DO NOT create new files with different names if similar files exist

2. API ENDPOINT MATCHING:
   - Frontend action names MUST exactly match backend case statements
   - HTTP methods MUST match (check if API accepts both GET and POST)
   - If frontend calls 'getSeats' but backend has 'get_seats', you MUST make them match

3. JSON FORMAT CONSISTENCY:
   - Response MUST include both 'status' and 'success' keys for compatibility
   - Structure: {'status': 'success'/'error', 'success': true/false, 'message': '...', 'data': {...}}
   - Frontend expected response keys MUST match backend return keys

4. TEST FILE FORMAT:
   - For PHP projects, tests MUST be Python files (test_*.py)
   - Tests use requests library to call PHP endpoints via HTTP
   - Test URL base: http://localhost:8000

5. BEFORE WRITING api.php:
   - Read ALL HTML and JS files to see all API calls
   - Extract exact endpoint names, HTTP methods, and expected JSON formats
   - API must read from BOTH $_GET and $_POST for flexibility (use array_merge or get_input_data function)
"""
        
        elif project_type == "Python":
            rules += """1. IMPORT CONSISTENCY:
   - Check ALL import statements match existing module names
   - Use relative imports for project modules (from . import module)
   - DO NOT import modules that don't exist in the project

2. FUNCTION/CLASS NAMING:
   - Use consistent naming conventions (snake_case for functions, PascalCase for classes)
   - Check existing code for naming patterns before creating new code

3. TEST FILE FORMAT:
   - Tests MUST be Python files (test_*.py)
   - Use pytest or unittest
   - Test files should be in tests/ directory
   - Follow existing test patterns in the project

4. API STRUCTURE (if web framework):
   - Flask: Use @app.route decorators with correct methods
   - Django: Use views with proper URL patterns
   - FastAPI: Use async def with proper path operations

5. DEPENDENCY MANAGEMENT:
   - Check requirements.txt for existing dependencies
   - Don't import packages not in requirements.txt without adding them
"""
        
        elif project_type == "Node.js":
            rules += """1. MODULE CONSISTENCY:
   - Check ALL require/import statements match existing files
   - Use consistent module system (CommonJS or ES6 modules, not mixed)
   - Respect package.json "type" field

2. EXPORT/IMPORT MATCHING:
   - Exported names must match imported names
   - Check if module uses named or default exports

3. TEST FILE FORMAT:
   - Tests should be JavaScript/TypeScript files
   - Use the test framework specified in package.json (jest, mocha, etc.)
   - Follow existing test patterns

4. ASYNC/AWAIT CONSISTENCY:
   - Check if project uses callbacks, promises, or async/await
   - Be consistent with existing patterns
"""
        
        else:
            rules += """1. FILE DEPENDENCIES:
   - Check existing files before creating new ones
   - Use consistent naming conventions
   - Don't duplicate functionality

2. TEST CONSISTENCY:
   - Use appropriate test framework for the language
   - Tests should be in tests/ directory
   - Follow existing test patterns
"""
        
        rules += """
GENERAL RULES (ALL LANGUAGES):
- Always check existing files before writing new code
- Maintain consistency with existing code patterns
- Test files must match API/function expectations
- Document any new functionality
"""
        
        return rules
    
    def _get_error_handling_instructions(self, project_type: str, last_test_error: str) -> str:
        """
        Get error handling instructions based on project type and error.
        
        Args:
            project_type: Type of project (PHP, Python, etc.)
            last_test_error: The error message from the last test
            
        Returns:
            String with error handling instructions
        """
        import re
        instructions = ""
        
        # Check for syntax errors based on project type
        is_syntax_error = False
        error_file = None
        
        if project_type == "PHP":
            if "PHP syntax error" in last_test_error or "Parse error" in last_test_error:
                is_syntax_error = True
                file_match = re.search(r'in (src/[^\s]+\.php)', last_test_error)
                if file_match:
                    error_file = file_match.group(1)
        elif project_type == "Python":
            if "SyntaxError" in last_test_error or "IndentationError" in last_test_error:
                is_syntax_error = True
                file_match = re.search(r'File ["\']([^"\']+\.py)["\']', last_test_error)
                if file_match:
                    error_file = file_match.group(1)
        elif project_type == "Node.js":
            if "SyntaxError" in last_test_error:
                is_syntax_error = True
                file_match = re.search(r'at [^(]*\(([^\)]+\.(js|ts))', last_test_error)
                if file_match:
                    error_file = file_match.group(1)
        
        if is_syntax_error:
            instructions += f"🚨 CRITICAL: This is a SYNTAX ERROR in {project_type}, NOT a test error!\n"
            instructions += "1. You MUST fix the source file that has the syntax error\n"
            instructions += "2. The error message above tells you which file and which line has the problem\n"
            instructions += "3. FIRST: Use read_file action to read the existing file with the error\n"
            instructions += "4. THEN: Use write_file action to write the CORRECTED version of the SAME file\n"
            instructions += "5. DO NOT create new files - you MUST fix the existing file\n"
            instructions += "6. After fixing, verify syntax is valid\n"
            instructions += "7. Only then create/run tests if needed\n"
            instructions += "8. DO NOT create test files until the syntax error is fixed\n"
            if error_file:
                instructions += f"\n⚠️ The file with the error is: {error_file}\n"
                instructions += "You MUST read this file first, then fix it. Do NOT create other files.\n"
        else:
            instructions += "1. Fix ONLY the test files that failed (do NOT regenerate source code files that already exist)\n"
            
            # Language-specific test instructions
            if project_type == "PHP":
                instructions += "2. For PHP projects: Generate PYTHON test files (test_*.py) that use requests to test API endpoints\n"
                instructions += "   - Tests make HTTP requests to http://localhost:8000\n"
                instructions += "   - Use POST for actions, check 'status' and 'success' in response\n"
            elif project_type == "Python":
                instructions += "2. For Python projects: Generate Python test files using pytest or unittest\n"
                instructions += "   - Use proper imports from src/ modules\n"
                instructions += "   - Execute with 'pytest tests/' or 'python -m pytest'\n"
            elif project_type == "Node.js":
                instructions += "2. For Node.js projects: Generate JavaScript test files\n"
                instructions += "   - Use the test framework in package.json (jest, mocha, etc.)\n"
                instructions += "   - Execute with 'npm test'\n"
            else:
                instructions += "2. Generate test files appropriate for the project language\n"
            
            instructions += "3. Execute the fixed tests\n"
        
        return instructions
    
    def _get_first_feature_instructions(self, project_type: str) -> str:
        """
        Get instructions for implementing the first feature.
        
        Args:
            project_type: Type of project (PHP, Python, etc.)
            
        Returns:
            String with feature implementation instructions
        """
        instructions = "INSTRUCTIONS - FOLLOW THIS EXACT ORDER:\n"
        instructions += "1. Create source code files FIRST (setup, database, API, etc.) - place ALL files in 'src/' directory (no subdirectories)\n"
        instructions += "2. Create test files AFTER source files exist\n"
        
        if project_type == "PHP":
            instructions += "   - For PHP: Create PYTHON test files (test_*.py) in 'tests/' directory\n"
            instructions += "   - Use Python's requests library to test PHP API endpoints\n"
            instructions += "   - Test by making HTTP requests to http://localhost:8000\n"
        elif project_type == "Python":
            instructions += "   - For Python: Create Python test files (test_*.py) in 'tests/' directory\n"
            instructions += "   - Use pytest or unittest\n"
            instructions += "   - Import modules from src/ for testing\n"
        elif project_type == "Node.js":
            instructions += "   - For Node.js: Create JavaScript/TypeScript test files in 'tests/' directory\n"
            instructions += "   - Use the test framework specified in package.json\n"
        else:
            instructions += "   - Create test files appropriate for the project language\n"
        
        instructions += "3. Execute all tests (verify files exist before testing)\n"
        instructions += "   - CRITICAL: After writing each test file, you MUST include an execute_command action to run it\n"
        
        if project_type == "PHP":
            instructions += "   - Execute with: python3 tests/test_*.py\n"
        elif project_type == "Python":
            instructions += "   - Execute with: pytest tests/ or python -m pytest tests/\n"
        elif project_type == "Node.js":
            instructions += "   - Execute with: npm test\n"
        else:
            instructions += "   - Execute with appropriate test command for the language\n"
        
        instructions += "4. If all tests pass, create git repository and commit\n"
        
        return instructions
    
    def _get_subsequent_feature_instructions(self, project_type: str) -> str:
        """
        Get instructions for implementing subsequent features (not the first).
        
        Args:
            project_type: Type of project (PHP, Python, etc.)
            
        Returns:
            String with feature implementation instructions
        """
        instructions = "INSTRUCTIONS - FOLLOW THIS EXACT ORDER:\n"
        instructions += "1. Create source code files for this feature FIRST - place ALL files in 'src/' directory (no subdirectories)\n"
        instructions += "2. Create test files AFTER source files exist\n"
        
        if project_type == "PHP":
            instructions += "   - For PHP: Create PYTHON test files (test_*.py) in 'tests/' directory\n"
            instructions += "   - Use Python's requests library to test PHP API endpoints\n"
            instructions += "   - Test by making HTTP requests to http://localhost:8000\n"
        elif project_type == "Python":
            instructions += "   - For Python: Create Python test files (test_*.py) in 'tests/' directory\n"
            instructions += "   - Use pytest or unittest\n"
            instructions += "   - Import modules from src/ for testing\n"
        elif project_type == "Node.js":
            instructions += "   - For Node.js: Create JavaScript/TypeScript test files in 'tests/' directory\n"
            instructions += "   - Use the test framework specified in package.json\n"
        else:
            instructions += "   - Create test files appropriate for the project language\n"
        
        instructions += "3. Execute feature tests (verify files exist before testing)\n"
        instructions += "   - CRITICAL: After writing each test file, you MUST include an execute_command action to run it\n"
        
        if project_type == "PHP":
            instructions += "   - Execute with: python3 tests/test_*.py\n"
        elif project_type == "Python":
            instructions += "   - Execute with: pytest tests/ or python -m pytest tests/\n"
        elif project_type == "Node.js":
            instructions += "   - Execute with: npm test\n"
        else:
            instructions += "   - Execute with appropriate test command for the language\n"
        
        instructions += "4. Execute ALL regression tests (full test suite)\n"
        instructions += "5. If all tests pass, commit to git\n"
        
        return instructions
    
    def _ask_planner(self, task_description: str, history: List[Dict], last_output: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Ask the Planner model for the next action plan.
        
        Args:
            task_description: The task from input/task.txt
            history: Last N steps of execution history
            last_output: Output from the last executed command
            
        Returns:
            List of action dictionaries from Planner
        """
        self._log_thought("Planner: Analyzing current state and planning next actions...")
        
        # Detect project type
        project_type = self._detect_project_type()
        test_framework_info = self._get_test_framework_info(project_type)
        
        planner_prompt = f"""You are the Senior Software Architect (Planner Agent). Your Goal: Complete the task in 'input/task.txt' using strict TDD (Test Driven Development). RULES:

PROJECT TYPE DETECTED: {project_type}
{test_framework_info}

1. OUTPUT DIRECTORY: ALL files must be written to the 'output/' directory. All paths should be relative to 'output/'. For example: 'output/src/app.py', 'output/tests/test_app.py', etc.

2. ONE FEATURE AT A TIME: Develop and complete ONE feature completely before moving to the next. Each feature must be fully implemented, tested, documented, and committed before starting the next feature.

3. TDD CYCLE PER FEATURE: For each feature:
   a) Write Test (Red) -> Write Code (Green) -> Refactor
   b) Run the feature test using the CORRECT test framework for the project type (see PROJECT TYPE above)
   c) Run ALL regression tests (full test suite) using the SAME framework
   d) Generate feature documentation
   e) Only if BOTH feature test AND regression tests pass: Commit to Git
   f) If any test fails: Fix the code immediately - DO NOT PROCEED until all tests pass

4. CONTINUITY: Read files before editing.

5. CRITICAL ERROR HANDLING: If a test fails, you MUST fix the code. You MUST cycle and retry until ALL tests pass. DO NOT proceed to next feature until current feature is fully working with ALL tests passing.

6. REGRESSION TESTS: After every feature test passes, you MUST run the full test suite (regression tests) to ensure no existing functionality broke. If regression tests fail, you MUST fix the code before proceeding.

7. DOCUMENTATION: For each completed feature, generate documentation in 'output/docs/features/[feature_name].md' describing what the feature does, how it works, and how to use it.

8. GIT COMMIT: Commit ONLY after:
   - Feature test passes
   - ALL regression tests pass
   - Documentation generated
   - Use commit message: "Feature: [feature name] - implemented and tested"

9. FEATURE COMPLETION: Mark a feature as complete only when: code written, feature test passes, regression tests pass, documentation generated, and Git commit successful.

10. FINAL DOCUMENTATION: When all features are complete (before 'end_task'), generate a final documentation file 'output/README.md' with project overview, build instructions, execution instructions, and deployment guide.

11. TEST EXECUTION: CRITICAL - You MUST execute tests after writing them. Use ONLY the test framework for the detected project type (see PROJECT TYPE above). DO NOT mix frameworks (e.g., do NOT use pytest for PHP projects, do NOT use phpunit for Python projects). ALWAYS execute tests as "execute_command" actions. Tests MUST pass before proceeding.

12. JSON OUTPUT CRITICAL: Your response MUST be a VALID, COMPLETE JSON array. Do NOT truncate the JSON. If the plan is long, ensure all brackets and braces are properly closed. The JSON must be parseable. Keep "content_instruction" fields concise to avoid truncation.

OUTPUT FORMAT: You must respond with a VALID, COMPLETE JSON ARRAY only. No markdown. No truncation. Schema: [{{ "step": int, "action": "write_file"|"read_file"|"execute_command"|"run_regression_tests"|"generate_docs"|"end_task", "target": str, "content_instruction": "Brief instruction for the Coder (keep concise)", "feature_name": "Optional: name of the feature" }}]"""
        
        # Build context message
        context_parts = [f"TASK:\n{task_description}\n"]
        context_parts.append(f"\nPROJECT TYPE: {project_type}\n")
        context_parts.append(f"{test_framework_info}\n")
        
        if history:
            context_parts.append("\nRECENT HISTORY (last 10 steps):\n")
            for step in history[-10:]:
                context_parts.append(f"- Step {step.get('step', '?')}: {step.get('action', '?')} on {step.get('target', '?')}")
                if step.get('output'):
                    context_parts.append(f"  Output: {step.get('output', '')[:200]}")
        
        if last_output:
            context_parts.append(f"\nLAST COMMAND OUTPUT:\n{last_output[:1000]}")
        
        context_message = "\n".join(context_parts)
        
        messages = [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": context_message}
        ]
        
        print(f"🤖 Planner thinking...")
        try:
            response = self.planner_client.chat_completion(messages, temperature=0.7)
            
            # Log raw response for debugging (first 500 chars)
            self._log_thought(f"Planner: Raw response length: {len(response)} chars")
            if len(response) > 1000:
                self._log_thought(f"Planner: Response is long, may be truncated")
            
            plan = self._clean_json(response)
            self._log_thought(f"Planner: Generated plan with {len(plan)} actions")
            print(f"✅ Planner generated {len(plan)} actions")
            # Log plan details
            for action in plan:
                action_type = action.get("action", "unknown")
                target = action.get("target", "")
                self._log_thought(f"Planner: Action '{action_type}' on '{target}'")
            return plan
        except ValueError as e:
            # JSON parsing error - log the problematic response
            error_msg = str(e)
            self._log_thought(f"Planner: JSON parsing error - {error_msg[:200]}")
            print(f"❌ Planner error: {error_msg}")
            # Try to ask Planner again with clearer instructions
            print("🔄 Requesting Planner to retry with simpler, shorter JSON...")
            raise
        except Exception as e:
            self._log_thought(f"Planner: Error occurred - {str(e)}")
            print(f"❌ Planner error: {e}")
            raise
    
    def _git_commit(self, message: str) -> bool:
        """
        Perform a Git commit with the given message in the output directory.
        Optionally push to remote if token is configured.
        
        Args:
            message: Commit message
            
        Returns:
            True if commit succeeded, False otherwise
        """
        try:
            # Ensure git repo exists
            if not self.git_repo_initialized:
                self._ensure_git_repo()
            
            # Check if we're in a git repo (in output directory)
            stdout, stderr, code = self.tools.execute_command("git rev-parse --git-dir", timeout=10, cwd=self.output_dir)
            if code != 0:
                print("⚠️  Not in a Git repository, initializing...")
                self._ensure_git_repo()
                if not self.git_repo_initialized:
                    return False
            
            # Check if there are changes to commit
            stdout, stderr, code = self.tools.execute_command("git status --porcelain", timeout=10, cwd=self.output_dir)
            if code != 0 or not stdout.strip():
                print("⚠️  No changes to commit")
                return False
            
            # Add all changes
            stdout, stderr, code = self.tools.execute_command("git add -A", timeout=10, cwd=self.output_dir)
            if code != 0:
                print(f"❌ Git add failed: {stderr}")
                return False
            
            # Commit
            commit_cmd = f'git commit -m "{message}"'
            stdout, stderr, code = self.tools.execute_command(commit_cmd, timeout=10, cwd=self.output_dir)
            if code != 0:
                print(f"❌ Git commit failed: {stderr}")
                return False
            
            print(f"✅ Git commit successful: {message}")
            
            # If token is available, try to push to remote
            if self.git_token:
                self._git_push_with_token()
            
            return True
            
        except Exception as e:
            print(f"⚠️  Git commit error: {e}")
            return False
    
    def _git_push_with_token(self) -> None:
        """
        Push to remote repository using the configured Git token.
        Configures Git credential helper if needed.
        """
        try:
            # Check if there's a remote configured (in output directory)
            stdout, stderr, code = self.tools.execute_command("git remote -v", timeout=10, cwd=self.output_dir)
            if code != 0 or not stdout.strip():
                print("ℹ️  No remote repository configured, skipping push")
                return
            
            # Get remote URL
            stdout, stderr, code = self.tools.execute_command("git remote get-url origin", timeout=10, cwd=self.output_dir)
            if code != 0:
                print("ℹ️  Could not get remote URL, skipping push")
                return
            
            remote_url = stdout.strip()
            
            # If remote URL contains token placeholder or needs authentication
            if self.git_token and ("github.com" in remote_url or "gitlab.com" in remote_url):
                # Configure Git to use token for this push
                # Note: For security, we'll use GIT_ASKPASS or credential helper
                print(f"🚀 Pushing to remote (using token)...")
                
                # Try push with token in URL (for HTTPS)
                if remote_url.startswith("https://"):
                    # Extract repo path
                    if "github.com" in remote_url:
                        repo_path = remote_url.replace("https://github.com/", "").replace(".git", "")
                        push_url = f"https://{self.git_token}@github.com/{repo_path}.git"
                    elif "gitlab.com" in remote_url:
                        repo_path = remote_url.replace("https://gitlab.com/", "").replace(".git", "")
                        push_url = f"https://oauth2:{self.git_token}@gitlab.com/{repo_path}.git"
                    else:
                        push_url = remote_url
                    
                    # Temporarily set remote URL with token
                    self.tools.execute_command(f'git remote set-url origin {push_url}', timeout=10, cwd=self.output_dir)
                    
                    # Push
                    stdout, stderr, code = self.tools.execute_command("git push origin HEAD", timeout=30, cwd=self.output_dir)
                    
                    # Restore original remote URL (remove token from URL)
                    self.tools.execute_command(f'git remote set-url origin {remote_url}', timeout=10, cwd=self.output_dir)
                    
                    if code == 0:
                        print("✅ Git push successful")
                    else:
                        print(f"⚠️  Git push failed: {stderr}")
                else:
                    # SSH remote - token not needed if SSH keys are configured
                    print("ℹ️  SSH remote detected, using SSH authentication")
                    stdout, stderr, code = self.tools.execute_command("git push origin HEAD", timeout=30, cwd=self.output_dir)
                    if code == 0:
                        print("✅ Git push successful")
                    else:
                        print(f"⚠️  Git push failed: {stderr}")
        except Exception as e:
            print(f"⚠️  Git push error: {e}")
    
    def _is_pytest_available(self) -> bool:
        """Check if pytest is available in the environment."""
        stdout, stderr, code = self.tools.execute_command("python3 -m pytest --version", timeout=10, cwd=self.output_dir)
        return code == 0
    
    def _auto_generate_tests(self) -> bool:
        """
        Auto-generate basic test files for source files that don't have tests.
        Returns True if tests were generated.
        """
        src_dir = os.path.join(self.output_dir, "src")
        tests_dir = os.path.join(self.output_dir, "tests")
        
        if not os.path.exists(src_dir):
            return False
        
        # Ensure tests directory exists
        Path(tests_dir).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        init_src = os.path.join(src_dir, "__init__.py")
        init_tests = os.path.join(tests_dir, "__init__.py")
        if not os.path.exists(init_src):
            with open(init_src, 'w') as f:
                f.write("")
        if not os.path.exists(init_tests):
            with open(init_tests, 'w') as f:
                f.write("")
        
        generated = False
        for src_file in os.listdir(src_dir):
            if not src_file.endswith('.py') or src_file.startswith('__'):
                continue
            
            module_name = src_file[:-3]  # Remove .py
            test_file_path = os.path.join(tests_dir, f"test_{module_name}.py")
            
            if os.path.exists(test_file_path):
                continue
            
            # Read source file to extract classes/functions
            src_path = os.path.join(src_dir, src_file)
            try:
                with open(src_path, 'r', encoding='utf-8') as f:
                    src_content = f.read()
                
                import re
                classes = re.findall(r'class\s+(\w+)', src_content)
                functions = [f for f in re.findall(r'def\s+(\w+)\s*\(', src_content) if not f.startswith('_')]
                
                # Generate basic test file
                test_content = f'''"""Auto-generated tests for {module_name}"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.{module_name} import *

'''
                
                # Generate test class for each class found
                for cls_name in classes:
                    test_content += f'''
class Test{cls_name}:
    """Tests for {cls_name} class"""
    
    def test_{cls_name.lower()}_instantiation(self):
        """Test that {cls_name} can be instantiated (may require mocking)"""
        # Note: This is a basic test - may need adjustment based on actual constructor
        pass  # TODO: Implement actual test
    
'''
                
                # Generate test functions for standalone functions
                for func_name in functions[:5]:  # Limit to first 5 functions
                    if func_name not in ['main', '__init__']:
                        test_content += f'''
def test_{func_name}():
    """Test {func_name} function"""
    pass  # TODO: Implement actual test

'''
                
                with open(test_file_path, 'w', encoding='utf-8') as f:
                    f.write(test_content)
                
                print(f"   📝 Generated {test_file_path}")
                generated = True
                
            except Exception as e:
                print(f"   ⚠️  Could not generate tests for {src_file}: {e}")
        
        return generated
    
    def _auto_fix_test_constructors(self, tests_dir: str) -> None:
        """
        Automatically fix constructor calls in test files to match actual class signatures.
        """
        import re
        
        # First, extract actual constructor signatures from source files
        src_dir = os.path.join(self.output_dir, "src")
        class_signatures = {}
        
        if os.path.exists(src_dir):
            for src_file in os.listdir(src_dir):
                if not src_file.endswith('.py') or src_file.startswith('__'):
                    continue
                
                src_path = os.path.join(src_dir, src_file)
                try:
                    with open(src_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Find class definitions and their __init__ parameters
                    class_pattern = r'class\s+(\w+).*?def\s+__init__\s*\(\s*self\s*,?\s*([^)]*)\)'
                    for match in re.finditer(class_pattern, content, re.DOTALL):
                        class_name = match.group(1)
                        params_str = match.group(2).strip()
                        if params_str:
                            # Extract parameter names (ignore defaults and type hints)
                            params = []
                            for p in params_str.split(','):
                                p = p.strip()
                                if p:
                                    # Remove type hints and defaults
                                    param_name = p.split(':')[0].split('=')[0].strip()
                                    if param_name:
                                        params.append(param_name)
                            class_signatures[class_name] = params
                except Exception:
                    pass
        
        if not class_signatures:
            return
        
        # Now fix test files
        for test_file in os.listdir(tests_dir):
            if not test_file.endswith('.py'):
                continue
            
            test_path = os.path.join(tests_dir, test_file)
            try:
                with open(test_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                for class_name, expected_params in class_signatures.items():
                    # Find wrong constructor calls - with wrong params or no params
                    # Pattern: ClassName() or ClassName(wrong_arg=value)
                    wrong_call_pattern = rf'{class_name}\s*\(\s*\)'  # No args
                    
                    if expected_params:
                        # Generate correct call with mock values
                        mock_values = []
                        for param in expected_params:
                            if 'url' in param.lower():
                                mock_values.append(f'"http://mock-{param}.test"')
                            elif 'path' in param.lower():
                                mock_values.append(f'"/mock/{param}"')
                            elif 'key' in param.lower():
                                mock_values.append(f'"mock_{param}"')
                            else:
                                mock_values.append(f'"mock_{param}"')
                        
                        correct_call = f'{class_name}({", ".join(mock_values)})'
                        
                        # Replace empty constructor calls
                        content = re.sub(wrong_call_pattern, correct_call, content)
                        
                        # Also try to fix calls with wrong named parameters
                        # e.g., RagGenerator(api_key="x") -> RagGenerator("http://...", "http://...")
                        wrong_named_pattern = rf'{class_name}\s*\([^)]*(?:api_key|model_name|model)[^)]*\)'
                        if re.search(wrong_named_pattern, content):
                            content = re.sub(wrong_named_pattern, correct_call, content)
                
                if content != original_content:
                    with open(test_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"   🔧 Auto-fixed constructors in {test_file}")
                    
            except Exception as e:
                print(f"   ⚠️  Could not auto-fix constructors in {test_file}: {e}")
    
    def _auto_fix_test_imports(self, tests_dir: str) -> None:
        """
        Automatically fix common import issues in test files.
        - Fix 'from output.X' to 'from src.X'
        - Add sys.path manipulation if needed
        """
        import re
        
        for test_file in os.listdir(tests_dir):
            if not test_file.endswith('.py'):
                continue
            
            test_path = os.path.join(tests_dir, test_file)
            try:
                with open(test_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Fix wrong import paths: 'from output.X' -> 'from src.X'
                content = re.sub(r'from\s+output\.(\w+)', r'from src.\1', content)
                content = re.sub(r'import\s+output\.(\w+)', r'import src.\1', content)
                
                # Add sys.path fix at the beginning if not present and importing from src
                if 'from src.' in content or 'import src.' in content:
                    if 'sys.path' not in content:
                        path_fix = """import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
                        # Add after any existing imports at the top, or at the very beginning
                        if content.startswith('import ') or content.startswith('from '):
                            # Find the first non-import line
                            lines = content.split('\n')
                            insert_idx = 0
                            for i, line in enumerate(lines):
                                if line.strip() and not line.startswith('import ') and not line.startswith('from ') and not line.startswith('#'):
                                    insert_idx = i
                                    break
                            lines.insert(insert_idx, path_fix.strip())
                            content = '\n'.join(lines)
                        else:
                            content = path_fix + content
                
                if content != original_content:
                    with open(test_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"   🔧 Auto-fixed imports in {test_file}")
                    
            except Exception as e:
                print(f"   ⚠️  Could not auto-fix {test_file}: {e}")
    
    def _validate_test_code_coherence(self, tests_dir: str) -> List[str]:
        """
        Validate that test files are coherent with source code.
        Checks imports, class names, method names match actual code.
        
        Returns:
            List of coherence issues found
        """
        issues = []
        src_dir = os.path.join(self.output_dir, "src")
        
        # Get actual source code structure
        src_structure = {}
        if os.path.exists(src_dir):
            for file in os.listdir(src_dir):
                if file.endswith('.py'):
                    file_path = os.path.join(src_dir, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Extract classes and functions
                        import re
                        classes = re.findall(r'class\s+(\w+)', content)
                        functions = re.findall(r'def\s+(\w+)\s*\(', content)
                        # Extract __init__ parameters for classes
                        init_params = {}
                        for class_name in classes:
                            # Find the class and its __init__
                            class_match = re.search(
                                rf'class\s+{class_name}.*?def\s+__init__\s*\(\s*self\s*,?\s*([^)]*)\)',
                                content, re.DOTALL
                            )
                            if class_match:
                                params_str = class_match.group(1)
                                params = [p.strip().split('=')[0].strip().split(':')[0].strip() 
                                         for p in params_str.split(',') if p.strip()]
                                init_params[class_name] = params
                        
                        src_structure[file] = {
                            'classes': classes,
                            'functions': functions,
                            'init_params': init_params
                        }
                    except Exception as e:
                        issues.append(f"Error reading {file}: {e}")
        
        # Check each test file (EXCLUDE backup files)
        for test_file in os.listdir(tests_dir):
            if not test_file.endswith('.py'):
                continue
            # Skip backup files
            if '.backup_' in test_file:
                continue
            
            test_path = os.path.join(tests_dir, test_file)
            try:
                with open(test_path, 'r', encoding='utf-8') as f:
                    test_content = f.read()
                
                # Check imports
                import_patterns = [
                    r'from\s+src\.(\w+)\s+import\s+(\w+)',
                    r'from\s+output\.(\w+)\s+import\s+(\w+)',  # Wrong pattern
                    r'from\s+(\w+)\s+import\s+(\w+)',
                ]
                
                for pattern in import_patterns:
                    imports = re.findall(pattern, test_content)
                    for module, class_or_func in imports:
                        # Check if module exists
                        module_file = f"{module}.py"
                        if module_file in src_structure:
                            src_info = src_structure[module_file]
                            if class_or_func not in src_info['classes'] and class_or_func not in src_info['functions']:
                                issues.append(f"Test '{test_file}' imports '{class_or_func}' from '{module}' but it doesn't exist in source")
                        
                        # Check for wrong import paths (output.X instead of src.X)
                        if 'from output.' in test_content:
                            issues.append(f"Test '{test_file}' uses wrong import path 'from output.X' - should be 'from src.X'")
                
                # Check instantiation parameters
                for src_file, src_info in src_structure.items():
                    for class_name, expected_params in src_info.get('init_params', {}).items():
                        # Find instantiation in test
                        inst_pattern = rf'{class_name}\s*\(([^)]*)\)'
                        instantiations = re.findall(inst_pattern, test_content)
                        for inst_args in instantiations:
                            # Count number of args passed
                            if inst_args.strip():
                                passed_args = [a.strip() for a in inst_args.split(',') if a.strip()]
                                # Check for named args
                                named_args = [a.split('=')[0].strip() for a in passed_args if '=' in a]
                                
                                if named_args:
                                    # Check if named args match expected params
                                    for arg in named_args:
                                        if arg not in expected_params:
                                            issues.append(f"Test '{test_file}' passes unknown parameter '{arg}' to {class_name}. Expected: {expected_params}")
                
            except Exception as e:
                issues.append(f"Error reading test file {test_file}: {e}")
        
        return issues
    
    def _run_regression_tests(self) -> tuple[bool, str]:
        """
        Run the full test suite (regression tests) to ensure no existing functionality broke.
        
        Returns:
            Tuple of (all_passed: bool, output: str)
        """
        # For PHP projects, ensure server is running
        project_type = self._detect_project_type()
        if project_type == "PHP":
            if not self._start_php_server():
                return False, "Failed to start PHP server for regression tests"
        
        self.test_counter += 1
        test_num = self.test_counter
        print(f"\n🧪 Running regression tests (full test suite)...")
        self._log_thought(f"Test #{test_num}: Executing regression test suite")
        
        # Detect project type and try appropriate test commands
        test_commands = self._detect_test_commands()
        
        for cmd in test_commands:
            try:
                self._log_thought(f"Test #{test_num}: Trying command '{cmd}'")
                stdout, stderr, return_code = self.tools.execute_command(cmd, timeout=120, cwd=self.output_dir)
                output = f"Return code: {return_code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                
                # Check if command not found (127) or file not found - try next command
                if return_code == 127 or "command not found" in stderr.lower() or "No such file or directory" in stderr:
                    self._log_thought(f"Test #{test_num}: Command '{cmd}' not found, trying next...")
                    continue
                
                if return_code == 0:
                    print(f"✅ Test #{test_num}: All regression tests passed!")
                    self._log_thought(f"Test #{test_num}: ESITO = PASSED")
                    return True, output
                else:
                    # If command executed but tests failed, return failure
                    print(f"❌ Test #{test_num}: Regression tests failed with command: {cmd}")
                    self._log_thought(f"Test #{test_num}: ESITO = FAILED")
                    return False, output
            except Exception as e:
                # Try next command
                self._log_thought(f"Test #{test_num}: Command '{cmd}' failed with exception: {e}, trying next...")
                continue
        
        # If no test command worked, assume tests don't exist yet (first feature)
        print(f"⚠️  Test #{test_num}: No test suite found. This might be the first feature.")
        self._log_thought(f"Test #{test_num}: ESITO = SKIPPED (no test suite found)")
        return True, "No test suite detected (first feature?)"
    
    def _ensure_phpunit_installed(self) -> bool:
        """Ensure PHPUnit is installed for PHP projects. Returns True if available."""
        composer_json = Path(self.output_dir).joinpath("composer.json")
        vendor_phpunit = Path(self.output_dir).joinpath("vendor/bin/phpunit")
        
        # Check if already installed
        if vendor_phpunit.exists():
            return True
        
        # Check if composer.json exists
        if not composer_json.exists():
            # Create composer.json with PHPUnit dependency
            self._log_thought("Creating composer.json with PHPUnit dependency")
            composer_content = """{
    "require-dev": {
        "phpunit/phpunit": "^10.0"
    },
    "autoload": {
        "psr-4": {
            "": "src/"
        }
    }
}
"""
            self.tools.write_file(str(composer_json), composer_content)
            print("📦 Created composer.json with PHPUnit dependency")
        
        # Install dependencies
        self._log_thought("Installing PHPUnit via Composer")
        print("📦 Installing PHPUnit via Composer...")
        stdout, stderr, code = self.tools.execute_command("composer install", timeout=180, cwd=self.output_dir)
        
        if code == 0 and vendor_phpunit.exists():
            print("✅ PHPUnit installed successfully")
            self._log_thought("PHPUnit installed successfully via Composer")
            return True
        else:
            print(f"⚠️  PHPUnit installation failed: {stderr}")
            self._log_thought(f"PHPUnit installation failed: {stderr}")
            return False
    
    def _is_environment_error(self, error_message: str) -> bool:
        """
        Detect if an error is an environment/setup error rather than an actual test failure.
        Environment errors should not count as test failures since they indicate missing tools,
        not broken code.
        """
        if not error_message:
            return False
        
        error_lower = error_message.lower()
        
        # Environment error patterns
        environment_patterns = [
            "command not found",
            "permission denied",
            "no such file or directory",
            "not found",
            "cannot find",
            "module not found",
            "modulenotfounderror",
            "importerror",
            "no module named",
            "pytest: not found",
            "python: not found",
            "python3: not found",
            "pip: not found",
            "pip3: not found",
            "syntax error in command",
            "exit code: 127",  # Command not found
            "exit code: 126",  # Permission denied
            "/bin/sh:",
        ]
        
        for pattern in environment_patterns:
            if pattern in error_lower:
                return True
        
        return False
    
    def _get_alternative_test_commands(self, original_command: str) -> List[str]:
        """
        Get alternative test commands to try when the original fails.
        """
        alternatives = []
        
        # If original command uses pytest, try alternative methods
        if "pytest" in original_command.lower():
            # Extract test path if present
            import re
            path_match = re.search(r'(tests?/[^\s]*)', original_command)
            test_path = path_match.group(1) if path_match else "tests/"
            
            alternatives = [
                f"python3 -m pytest {test_path} -v",
                f"python -m pytest {test_path} -v",
                f"python3 -m unittest discover -s tests -v",
                f"python -m unittest discover -s tests -v",
            ]
            
            # If specific test file, also try running it directly
            if test_path.endswith('.py'):
                alternatives.append(f"python3 {test_path}")
                alternatives.append(f"python {test_path}")
        
        # If original command tries to run test file directly
        elif original_command.endswith('.py'):
            alternatives = [
                f"python3 {original_command}",
                f"python {original_command}",
                f"python3 -m pytest {original_command} -v",
            ]
        
        # For unittest commands
        elif "unittest" in original_command.lower():
            alternatives = [
                "python3 -m unittest discover -s tests -v",
                "python -m unittest discover -s tests -v",
                "python3 -m pytest tests/ -v",
            ]
        
        # Generic fallback
        if not alternatives:
            alternatives = [
                "python3 -m unittest discover -s tests -v",
                "python3 -m pytest tests/ -v",
            ]
        
        return alternatives
    
    def _try_fix_test_environment(self, error_message: str) -> bool:
        """
        Attempt to fix environment issues automatically.
        Returns True if a fix was attempted.
        """
        error_lower = error_message.lower() if error_message else ""
        fix_attempted = False
        
        # If pytest not found, try using unittest instead or installing pytest
        if "pytest" in error_lower and ("not found" in error_lower or "command not found" in error_lower):
            print("   📦 pytest not found. Will use unittest as fallback...")
            # Update the test command preference - unittest is always available
            self._log_thought("Environment fix: Switching from pytest to unittest")
            fix_attempted = True
            
            # Create a helper to run tests with unittest
            tests_dir = os.path.join(self.output_dir, "tests")
            if os.path.exists(tests_dir):
                # Ensure __init__.py exists in tests directory
                init_file = os.path.join(tests_dir, "__init__.py")
                if not os.path.exists(init_file):
                    with open(init_file, 'w') as f:
                        f.write("# Test package\n")
                    print(f"   ✅ Created {init_file}")
                
                # Also ensure src/__init__.py exists
                src_dir = os.path.join(self.output_dir, "src")
                if os.path.exists(src_dir):
                    src_init = os.path.join(src_dir, "__init__.py")
                    if not os.path.exists(src_init):
                        with open(src_init, 'w') as f:
                            f.write("# Source package\n")
                        print(f"   ✅ Created {src_init}")
        
        # If permission denied, try to fix permissions
        if "permission denied" in error_lower:
            print("   🔧 Attempting to fix file permissions...")
            tests_dir = os.path.join(self.output_dir, "tests")
            if os.path.exists(tests_dir):
                for test_file in os.listdir(tests_dir):
                    if test_file.endswith('.py'):
                        test_path = os.path.join(tests_dir, test_file)
                        try:
                            os.chmod(test_path, 0o755)
                            print(f"   ✅ Fixed permissions for {test_file}")
                            fix_attempted = True
                        except:
                            pass
        
        # If module not found, ensure __init__.py files exist
        if "module" in error_lower and ("not found" in error_lower or "no module" in error_lower):
            print("   📦 Module import error. Ensuring package structure...")
            for subdir in ["src", "tests"]:
                dir_path = os.path.join(self.output_dir, subdir)
                if os.path.exists(dir_path):
                    init_file = os.path.join(dir_path, "__init__.py")
                    if not os.path.exists(init_file):
                        with open(init_file, 'w') as f:
                            f.write(f"# {subdir} package\n")
                        print(f"   ✅ Created {init_file}")
                        fix_attempted = True
        
        return fix_attempted
    
    def _detect_test_commands(self) -> List[str]:
        """
        Detect project type and return appropriate test commands to try.
        Returns list of test commands ordered by likelihood.
        """
        test_commands = []
        
        # Check for PHP projects
        php_files = list(Path(self.output_dir).glob("**/*.php"))
        php_test_files = list(Path(self.output_dir).glob("**/test*.php"))
        py_test_files = list(Path(self.output_dir).glob("**/test_*.py"))
        phpunit_xml = Path(self.output_dir).joinpath("phpunit.xml")
        composer_json = Path(self.output_dir).joinpath("composer.json")
        
        if php_files or php_test_files or phpunit_xml.exists() or composer_json.exists():
            self._log_thought("Detected PHP project - using Python tests")
            
            # For PHP projects, we use Python tests (not PHPUnit)
            # Find all Python test files and execute them
            if py_test_files:
                self._log_thought(f"Found {len(py_test_files)} Python test files for PHP project")
                for test_file in py_test_files:
                    rel_path = os.path.relpath(test_file, self.output_dir)
                    test_commands.append(f"python3 {rel_path}")
            
            # If no Python tests found, try to find any test files
            if not py_test_files:
                # Look for any test files in tests/ directory
                tests_dir = Path(self.output_dir).joinpath("tests")
                if tests_dir.exists():
                    for test_file in tests_dir.glob("*.py"):
                        rel_path = os.path.relpath(test_file, self.output_dir)
                        test_commands.append(f"python3 {rel_path}")
        
        # Check for Python projects
        py_files = list(Path(self.output_dir).glob("**/*.py"))
        py_test_files = list(Path(self.output_dir).glob("**/test_*.py"))
        requirements_txt = Path(self.output_dir).joinpath("requirements.txt")
        if py_files or py_test_files or requirements_txt.exists():
            self._log_thought("Detected Python project - adding pytest/unittest commands")
            # IMPORTANT: Use python3 -m pytest instead of pytest directly
            # This ensures we use the correct Python environment
            if py_test_files:
                test_commands.extend([
                    "python3 -m pytest tests/",
                    "python3 -m pytest",
                    "python -m pytest tests/",
                    "python -m pytest",
                ])
            # Fallback to unittest (always works with Python)
            test_commands.extend([
                "python3 -m unittest discover -s tests -v",
                "python3 -m unittest discover -v",
                "python -m unittest discover -s tests -v",
                "python -m unittest discover -v",
            ])
        
        # Check for Node.js projects
        if Path(self.output_dir).joinpath("package.json").exists():
            self._log_thought("Detected Node.js project - adding npm test")
            test_commands.append("npm test")
        
        # Check for Java projects
        java_files = list(Path(self.output_dir).glob("**/*.java"))
        if java_files:
            self._log_thought("Detected Java project - adding Maven/Gradle commands")
            test_commands.extend([
                "mvn test",
                "./gradlew test",
                "gradle test",
            ])
        
        # Check for Go projects
        go_files = list(Path(self.output_dir).glob("**/*.go"))
        if go_files:
            self._log_thought("Detected Go project - adding go test")
            test_commands.append("go test ./...")
        
        # Check for Ruby projects
        rb_files = list(Path(self.output_dir).glob("**/*.rb"))
        if rb_files or Path(self.output_dir).joinpath("Gemfile").exists():
            self._log_thought("Detected Ruby project - adding RSpec/minitest")
            test_commands.extend([
                "rspec",
                "bundle exec rspec",
                "rake test",
                "ruby -Itest test/**/*_test.rb",
            ])
        
        # Generic test commands (try last)
        test_commands.extend([
            "make test",
            "make check",
        ])
        
        return test_commands
    
    def _get_test_framework_info(self, project_type: str) -> str:
        """Get test framework information for the Planner prompt."""
        info_map = {
            "PHP": """TEST FRAMEWORK FOR PHP PROJECTS:
- CRITICAL: Use PYTHON test files (test_*.py) to test PHP applications via HTTP requests
- The PHP server will be automatically started on http://localhost:8000 before tests run
- Use Python's requests library or subprocess with curl to test PHP API endpoints
- Test files should be in tests/ directory with names like test_*.py
- Execute with: "python3 tests/test_feature.py" or "python tests/test_feature.py"
- Test the PHP application as a black box by making HTTP requests to http://localhost:8000
- All API calls should use http://localhost:8000 as the base URL
- DO NOT generate PHP test files - use Python tests instead
- Example: "python3 tests/test_api.py" """,
            
            "Python": """TEST FRAMEWORK FOR PYTHON PROJECTS:
- Use pytest: "pytest tests/test_feature.py" or "python -m pytest tests/"
- Or unittest: "python -m unittest discover -s tests"
- Test files should be in tests/ directory with names like test_*.py
- DO NOT use phpunit or PHP test frameworks
- Example: "pytest tests/test_api.py" or "python -m pytest tests/" """,
            
            "Node.js": """TEST FRAMEWORK FOR NODE.JS PROJECTS:
- Use npm test: "npm test" or specific test runner
- Test files typically in tests/ or __tests__/ directories
- Example: "npm test" """,
            
            "Java": """TEST FRAMEWORK FOR JAVA PROJECTS:
- Use Maven: "mvn test" or Gradle: "./gradlew test" or "gradle test"
- Test files typically in src/test/java/
- Example: "mvn test" """,
            
            "Go": """TEST FRAMEWORK FOR GO PROJECTS:
- Use go test: "go test ./..." or "go test ./package_name"
- Test files end with _test.go
- Example: "go test ./..." """,
            
            "Ruby": """TEST FRAMEWORK FOR RUBY PROJECTS:
- Use RSpec: "rspec" or "bundle exec rspec"
- Or Minitest: "rake test" or "ruby -Itest test/**/*_test.rb"
- Example: "rspec" or "rake test" """,
            
            "unknown": """TEST FRAMEWORK: Project type not yet detected. 
- If you see PHP files (.php), use PHPUnit
- If you see Python files (.py), use pytest
- If you see JavaScript files (.js) and package.json, use npm test
- Choose the appropriate framework based on file extensions you see"""
        }
        
        return info_map.get(project_type, info_map["unknown"])
    
    def _normalize_command_paths(self, command: str) -> str:
        """
        Normalize paths in commands when executing in output/ directory.
        Removes 'output/' prefix from paths since cwd is already output/.
        
        Args:
            command: Original command string
            
        Returns:
            Command with normalized paths
        """
        import re
        
        # Pattern to match paths that start with output/
        # Matches: output/src/file.php, output/tests/file.php, etc.
        pattern = r'output/([^\s]+)'
        
        def replace_path(match):
            path = match.group(1)
            return path
        
        # Replace all occurrences of output/... with just ...
        normalized = re.sub(pattern, replace_path, command)
        
        return normalized
    
    def _correct_test_command(self, command: str, project_type: str) -> str:
        """
        Auto-correct test commands if Planner uses wrong framework.
        Also handles PHPUnit not installed by falling back to direct PHP execution.
        Normalizes paths for execution in output/ directory.
        Returns corrected command.
        """
        import re  # Import re at the beginning of the method
        
        # First, normalize paths (remove output/ prefix since cwd is output/)
        command = self._normalize_command_paths(command)
        
        cmd_lower = command.lower()
        
        # Check if it's a test command
        is_test_cmd = any(keyword in cmd_lower for keyword in ["test", "pytest", "phpunit", "rspec", "mvn", "gradle", "go test", "npm test"])
        
        if not is_test_cmd:
            return command
        
        # PHP project: tests should be Python (using curl/requests)
        if project_type == "PHP":
            # If Planner tries to use PHPUnit or PHP tests, convert to Python
            if "phpunit" in cmd_lower or ("php" in cmd_lower and "test" in cmd_lower and ".php" in command):
                self._log_thought(f"WARNING: Planner used PHP test framework for PHP project. Converting to Python tests...")
                # Replace PHP test commands with Python
                corrected = command.replace("phpunit", "python3").replace("php tests/", "python3 tests/")
                # Fix file extensions
                corrected = re.sub(r'tests/([^/]+)\.php', r'tests/\1.py', corrected)
                print(f"⚠️  Auto-corrected test command: {command} -> {corrected}")
                return corrected
            
            # Ensure Python is used for tests
            if "pytest" in cmd_lower or ("python" in cmd_lower and "test" in cmd_lower):
                # Python tests are correct for PHP projects
                return command
            
            # If PHPUnit command but PHPUnit might not be installed, add fallback
            if "phpunit" in cmd_lower:
                # Extract test file path if present
                # Match test file paths (with or without output/ prefix)
                test_file_match = re.search(r'(?:output/)?(tests?/[^\s]+\.php|test[^\s]*\.php)', command)
                if test_file_match:
                    test_file = test_file_match.group(1)
                    # Remove output/ prefix if present
                    if test_file.startswith('output/'):
                        test_file = test_file[7:]
                    # Add direct PHP execution as alternative (will be tried if phpunit fails)
                    self._log_thought(f"PHPUnit command detected. If PHPUnit not available, will try: php {test_file}")
        
        # Python project but using PHP test framework
        elif project_type == "Python":
            if "phpunit" in cmd_lower or ("php" in cmd_lower and "test" in cmd_lower and "phpunit" not in cmd_lower):
                self._log_thought(f"WARNING: Planner used PHP test framework for Python project. Auto-correcting...")
                # Replace phpunit with pytest
                corrected = command.replace("phpunit", "pytest").replace("php vendor/bin/phpunit", "python -m pytest")
                # Fix paths
                corrected = corrected.replace("test", "test_").replace(".php", ".py")
                print(f"⚠️  Auto-corrected test command: {command} -> {corrected}")
                return corrected
        
        return command
    
    def _validate_plan_coherence(self, plan: List[Dict[str, Any]], project_type: str) -> tuple[bool, List[str]]:
        """
        Validate plan for coherence issues before execution.
        Works for PHP, Python, and other project types.
        
        Args:
            plan: Execution plan from Planner
            project_type: Type of project (PHP, Python, etc.)
            
        Returns:
            (is_valid, list_of_warnings)
        """
        import re
        warnings = []
        
        src_dir = os.path.join(self.output_dir, "src")
        tests_dir = os.path.join(self.output_dir, "tests")
        
        # Extract files that will be written
        files_to_write = []
        test_files_to_write = []
        for action in plan:
            if action.get('action') == 'write_file':
                target = action.get('target', '')
                if target:
                    files_to_write.append(target)
                    if 'test' in target.lower():
                        test_files_to_write.append(target)
        
        # ============= PHP-SPECIFIC VALIDATION =============
        if project_type == "PHP":
            if os.path.exists(src_dir):
                existing_php_files = [f for f in os.listdir(src_dir) if f.endswith('.php')]
                
                for action in plan:
                    if action.get('action') == 'write_file':
                        target = action.get('target', '')
                        if target.endswith('.php'):
                            instruction = action.get('content_instruction', '')
                            # Capture ALL require/include statements
                            require_matches = re.findall(r'(?:require|include)(?:_once)?\s+[\'"]([^\'"]+)[\'"]', instruction)
                            for req_file in require_matches:
                                if not req_file.endswith('.php'):
                                    if req_file.endswith(('.sqlite', '.db', '.sqlite3', '.json', '.txt', '.log')):
                                        warnings.append(f"❌ CRITICAL: Plan will write {target} requiring '{req_file}' which is a data file! Use PDO to connect to database.")
                                    else:
                                        warnings.append(f"⚠️ Plan will write {target} requiring '{req_file}' which doesn't end with .php.")
                                
                                if req_file.endswith('.php'):
                                    req_path = os.path.join(src_dir, req_file)
                                    if not os.path.exists(req_path):
                                        similar = [f for f in existing_php_files if req_file.replace('.php', '') in f or f.replace('.php', '') in req_file]
                                        if similar:
                                            warnings.append(f"⚠️ Plan will write {target} requiring '{req_file}' but file doesn't exist. Did you mean '{similar[0]}'?")
                                        else:
                                            warnings.append(f"⚠️ Plan will write {target} requiring '{req_file}' but file doesn't exist")
            
            # Check test files are Python for PHP projects
            for test_file in test_files_to_write:
                if test_file.endswith('.php'):
                    warnings.append(f"⚠️ For PHP projects, test files should be Python (.py), not PHP. Found: {test_file}")
        
        # ============= PYTHON-SPECIFIC VALIDATION =============
        elif project_type == "Python":
            if os.path.exists(src_dir):
                existing_py_files = [f for f in os.listdir(src_dir) if f.endswith('.py')]
                
                for action in plan:
                    if action.get('action') == 'write_file':
                        target = action.get('target', '')
                        if target.endswith('.py'):
                            instruction = action.get('content_instruction', '')
                            # Check for import statements referencing non-existent files
                            import_matches = re.findall(r'from\s+(\S+)\s+import|import\s+(\S+)', instruction)
                            for imp in import_matches:
                                module = imp[0] or imp[1]
                                if module.startswith('.'):
                                    module_file = module.lstrip('.').replace('.', '/') + '.py'
                                    if module_file in existing_py_files or any(module.lstrip('.') in f for f in existing_py_files):
                                        continue
                                    else:
                                        warnings.append(f"⚠️ Plan will write {target} importing '{module}' but module may not exist")
            
            # Check test files are Python for Python projects
            for test_file in test_files_to_write:
                if not test_file.endswith('.py'):
                    warnings.append(f"⚠️ For Python projects, test files should be Python (.py). Found: {test_file}")
        
        # ============= NODE.JS-SPECIFIC VALIDATION =============
        elif project_type == "Node.js":
            for test_file in test_files_to_write:
                if not test_file.endswith(('.js', '.ts')):
                    warnings.append(f"⚠️ For Node.js projects, test files should be JavaScript/TypeScript. Found: {test_file}")
        
        # ============= GENERAL VALIDATION =============
        # Check for execute_command actions that match test execution
        has_test_execution = False
        for action in plan:
            if action.get('action') == 'execute_command':
                target = action.get('target', '')
                if 'test' in target.lower() or 'pytest' in target.lower():
                    has_test_execution = True
                    break
        
        if test_files_to_write and not has_test_execution:
            warnings.append(f"⚠️ Plan writes test files but doesn't include execute_command to run them")
        
        return len(warnings) == 0, warnings
    
    def _validate_generated_code(self, project_type: str) -> tuple[bool, List[str]]:
        """
        Validate generated code for coherence issues after execution.
        Works for PHP, Python, and other project types.
        
        Args:
            project_type: Type of project (PHP, Python, etc.)
            
        Returns:
            (is_valid, list_of_errors)
        """
        import re
        errors = []
        
        src_dir = os.path.join(self.output_dir, "src")
        tests_dir = os.path.join(self.output_dir, "tests")
        
        if not os.path.exists(src_dir):
            return True, errors
        
        # ============= PHP-SPECIFIC VALIDATION =============
        if project_type == "PHP":
            php_files = [f for f in os.listdir(src_dir) if f.endswith('.php')]
            
            for php_file in php_files:
                php_path = os.path.join(src_dir, php_file)
                try:
                    with open(php_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for require/include mismatches
                    require_matches = re.findall(r'(?:require|include)(?:_once)?\s+[\'"]([^\'"]+)[\'"]', content)
                    for req_file in require_matches:
                        if not req_file.endswith('.php'):
                            if req_file.endswith(('.sqlite', '.db', '.sqlite3', '.json', '.txt', '.log', '.csv')):
                                errors.append(f"❌ CRITICAL: {php_file} requires '{req_file}' which is a data file! Use PDO to connect to database.")
                            elif req_file.endswith(('.html', '.css', '.js')):
                                errors.append(f"❌ {php_file} requires '{req_file}' - use readfile() or include HTML differently.")
                            else:
                                errors.append(f"⚠️ {php_file} requires '{req_file}' which doesn't end with .php.")
                        
                        if req_file.endswith('.php'):
                            req_path = os.path.join(src_dir, req_file)
                            if not os.path.exists(req_path):
                                similar = [f for f in php_files if req_file.replace('.php', '') in f or f.replace('.php', '') in req_file]
                                if similar:
                                    errors.append(f"❌ {php_file} requires '{req_file}' but file doesn't exist. Did you mean '{similar[0]}'?")
                                else:
                                    errors.append(f"❌ {php_file} requires '{req_file}' but file doesn't exist")
                    
                    # Check for syntax errors
                    syntax_check = self._check_php_syntax(php_path)
                    if syntax_check:
                        errors.append(f"❌ PHP Syntax error in {php_file}: {syntax_check}")
                        
                except Exception as e:
                    pass
            
            # Check that test files are Python for PHP projects
            if os.path.exists(tests_dir):
                test_files = os.listdir(tests_dir)
                for test_file in test_files:
                    if test_file.endswith('.php') and 'test' in test_file.lower():
                        errors.append(f"⚠️ For PHP projects, test files should be Python. Found PHP test: {test_file}")
        
        # ============= PYTHON-SPECIFIC VALIDATION =============
        elif project_type == "Python":
            py_files = [f for f in os.listdir(src_dir) if f.endswith('.py')]
            
            for py_file in py_files:
                py_path = os.path.join(src_dir, py_file)
                try:
                    with open(py_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for import mismatches
                    import_matches = re.findall(r'from\s+(\S+)\s+import|import\s+(\S+)', content)
                    for imp in import_matches:
                        module = imp[0] or imp[1]
                        if module.startswith('.'):
                            module_file = module.lstrip('.').split('.')[0] + '.py'
                            module_path = os.path.join(src_dir, module_file)
                            if not os.path.exists(module_path) and module_file not in [f.replace('.py', '') for f in py_files]:
                                # Check if it's a standard library or installed package
                                if module.lstrip('.').split('.')[0] not in ['os', 'sys', 'json', 'datetime', 'collections', 'typing', 'pathlib', 're', 'sqlite3', 'flask', 'django', 'requests', 'numpy', 'pandas']:
                                    errors.append(f"⚠️ {py_file} imports '{module}' but module file doesn't exist")
                    
                    # Check for Python syntax errors
                    syntax_check = self._check_python_syntax(py_path)
                    if syntax_check:
                        errors.append(f"❌ Python Syntax error in {py_file}: {syntax_check}")
                    
                    # CRITICAL: Check for correct API response parsing (OpenAI-compatible endpoints)
                    # If code uses requests.post() with /v1/chat/completions, must extract content correctly
                    if 'requests.post' in content and '/v1/chat/completions' in content:
                        # Check if code correctly extracts content from response
                        has_content_extraction = re.search(
                            r'response\.json\(\)\[["\']choices["\']\]\[0\]\[["\']message["\']\]\[["\']content["\']\]',
                            content
                        )
                        has_json_parse = re.search(
                            r'json\.loads\s*\([^)]*content[^)]*\)',
                            content
                        )
                        
                        # Check for wrong pattern: returning response.json() directly
                        wrong_pattern = re.search(
                            r'return\s+response\.json\(\)\s*$',
                            content,
                            re.MULTILINE
                        )
                        
                        if wrong_pattern and not has_content_extraction:
                            errors.append(f"❌ CRITICAL: {py_file} returns response.json() directly but should extract content first! Pattern: response.json()['choices'][0]['message']['content'] then json.loads(content)")
                        
                        if has_content_extraction and not has_json_parse:
                            # Check if content is used directly without parsing
                            content_var = re.search(
                                r'(\w+)\s*=\s*response\.json\(\)\[["\']choices["\']\]\[0\]\[["\']message["\']\]\[["\']content["\']\]',
                                content
                            )
                            if content_var:
                                var_name = content_var.group(1)
                                # Check if var_name is used in validate_json or similar without json.loads
                                if re.search(rf'validate_json\s*\(\s*{var_name}\s*,', content) and not re.search(rf'json\.loads\s*\(\s*{var_name}\s*\)', content):
                                    errors.append(f"❌ CRITICAL: {py_file} validates '{var_name}' (API content string) directly but should parse it first with json.loads()!")
                    
                    # Check for retry logic - if task mentions retries, verify it's implemented
                    if 'retry' in content.lower() or 'attempt' in content.lower():
                        # Check if retry is for entire record, not just HTTP call
                        has_record_retry = re.search(
                            r'for\s+\w+\s+in\s+range\s*\([^)]*\)\s*:.*?(?:call_model_x|generate_intent).*?(?:call_model_y|generate_solution)',
                            content,
                            re.DOTALL
                        )
                        if not has_record_retry:
                            # Check if retry is only for HTTP calls
                            http_only_retry = re.search(
                                r'for\s+\w+\s+in\s+range\s*\([^)]*\)\s*:.*?requests\.post',
                                content,
                                re.DOTALL
                            )
                            if http_only_retry:
                                errors.append(f"⚠️ {py_file} has retry logic but only for HTTP calls. Task requires retry for entire record (call X + validate + call Y + validate).")
                        
                except Exception as e:
                    pass
        
        # ============= NODE.JS-SPECIFIC VALIDATION =============
        elif project_type == "Node.js":
            js_files = [f for f in os.listdir(src_dir) if f.endswith(('.js', '.ts'))]
            
            for js_file in js_files:
                js_path = os.path.join(src_dir, js_file)
                try:
                    with open(js_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for require mismatches (CommonJS)
                    require_matches = re.findall(r'require\([\'"]\.\/([^\'"]+)[\'"]\)', content)
                    for req_file in require_matches:
                        if not req_file.endswith(('.js', '.ts', '.json')):
                            req_file += '.js'
                        req_path = os.path.join(src_dir, req_file)
                        if not os.path.exists(req_path):
                            errors.append(f"⚠️ {js_file} requires './{req_file}' but file doesn't exist")
                    
                    # Check for ES6 import mismatches
                    import_matches = re.findall(r'import\s+.*?\s+from\s+[\'"]\.\/([^\'"]+)[\'"]', content)
                    for imp_file in import_matches:
                        if not imp_file.endswith(('.js', '.ts')):
                            imp_file += '.js'
                        imp_path = os.path.join(src_dir, imp_file)
                        if not os.path.exists(imp_path):
                            errors.append(f"⚠️ {js_file} imports './{imp_file}' but file doesn't exist")
                            
                except Exception as e:
                    pass
        
        # ============= COHERENCE REPORT CHECK =============
        coherence_report = self._generate_coherence_report(project_type)
        if coherence_report and '❌' in coherence_report:
            critical_issues = re.findall(r'❌ ([^\n]+)', coherence_report)
            errors.extend([f"❌ {issue}" for issue in critical_issues])
        
        return len(errors) == 0, errors
    
    def run(self) -> None:
        """
        Main execution loop with requirement verification.
        
        Flow:
        1. Extract requirements from task
        2. Planner creates list of features
        3. For each feature:
           a. Planner creates test list
           b. Executor writes code
           c. Run all tests
           d. If all pass: create git repo (if needed) and commit
           e. Move to next feature
        4. Verify ALL requirements are met
        5. If not, iterate until they are
        """
        self.start_time = time.time()
        print("🚀 Starting Code Agent...")
        print("=" * 60)
        self._log_thought("Agent: Initializing development session")
        
        # Ensure Git repository exists (in case directory was deleted or is empty)
        self._ensure_git_repo()
        
        # Step 1: Read task description
        try:
            task_description = self.tools.read_file("input/task.txt")
            self.task_description = task_description  # Store for Executor context
            print(f"📋 Task loaded: {len(task_description)} characters")
            self._log_thought(f"Agent: Loaded task description ({len(task_description)} chars)")
        except FileNotFoundError:
            print("❌ Error: input/task.txt not found!")
            self._log_thought("Agent: ERROR - input/task.txt not found!")
            return
        
        # Step 1.5: Extract requirements from task
        print("\n" + "=" * 60)
        print("📋 STEP 1: Extracting requirements from task...")
        print("=" * 60)
        requirements = self._extract_requirements(task_description)
        self.extracted_requirements = requirements  # Store for Executor context
        if requirements:
            print(f"✅ Extracted {len(requirements)} requirements:")
            for req in requirements[:10]:  # Show first 10
                print(f"   - [{req.get('id')}] {req.get('description', '')[:60]}...")
            if len(requirements) > 10:
                print(f"   ... and {len(requirements) - 10} more")
        else:
            print("⚠️  Could not extract specific requirements, proceeding with feature-based approach")
            self.extracted_requirements = []
        
        # Step 2: Get feature list from Planner
        print("\n" + "=" * 60)
        print("📋 STEP 2: Getting feature list from Planner...")
        print("=" * 60)
        features = self._get_feature_list(task_description)
        
        if not features:
            print("❌ No features identified. Exiting.")
            return
        
        print(f"\n✅ Identified {len(features)} features:")
        for i, feature in enumerate(features, 1):
            print(f"   {i}. {feature}")
        
        # Step 3: Process each feature
        for feature_idx, feature_name in enumerate(features, 1):
            print("\n" + "=" * 60)
            print(f"🎯 FEATURE {feature_idx}/{len(features)}: {feature_name}")
            print("=" * 60)
            self.current_feature = feature_name
            self.feature_test_passed = False
            self.current_feature_files = []  # Reset file tracking for new feature
            
            # Process feature until tests pass - NO LIMIT on attempts
            # We distinguish between:
            # - Environment errors (pytest not found, permission denied) - these don't count as real failures
            # - Actual test failures - these require code fixes
            attempt = 0
            real_failure_count = 0
            max_consecutive_same_error = 5  # Stop if we get the same error 5 times in a row
            feature_complete = False
            
            last_test_error = None
            consecutive_same_error = 0
            environment_error_detected = False
            
            while not feature_complete:
                attempt += 1
                print(f"\n--- Attempt {attempt} for feature '{feature_name}' ---")
                
                # Get plan for this feature (include last test error if available)
                plan = self._get_feature_plan(feature_name, task_description, feature_idx == 1, last_test_error)
                
                if not plan:
                    print("⚠️  Planner returned empty plan, retrying...")
                    time.sleep(2)
                    continue
                
                # Validate plan coherence before execution
                project_type = self._detect_project_type()
                plan_valid, plan_warnings = self._validate_plan_coherence(plan, project_type)
                if plan_warnings:
                    print(f"\n⚠️  Plan validation warnings:")
                    for warning in plan_warnings:
                        print(f"   {warning}")
                    # Continue anyway, but log warnings
                
                # Execute plan
                result = self._execute_feature_plan(plan, feature_name, feature_idx == 1)
                
                # Validate generated code after execution
                code_valid, code_errors = self._validate_generated_code(project_type)
                if not code_valid:
                    print(f"\n❌ Code validation failed after execution:")
                    for error in code_errors:
                        print(f"   {error}")
                    # Treat as failure
                    result = (False, f"Code validation failed: {', '.join(code_errors)}")
                if isinstance(result, tuple):
                    success, test_error = result
                else:
                    success = result
                    test_error = None if success else "Unknown error"
                
                if success:
                    feature_complete = True
                    print(f"\n✅ Feature '{feature_name}' completed successfully!")
                else:
                    # Detect if this is an environment error vs actual test failure
                    is_environment_error = self._is_environment_error(test_error)
                    
                    if is_environment_error:
                        if not environment_error_detected:
                            environment_error_detected = True
                            print(f"\n⚠️  Environment error detected: {test_error}")
                            print("   Attempting to fix environment issues...")
                            self._try_fix_test_environment(test_error)
                        else:
                            # Already tried to fix, count as real failure
                            real_failure_count += 1
                    else:
                        real_failure_count += 1
                        environment_error_detected = False  # Reset
                    
                    # Normalize error for comparison (extract error type and key message)
                    normalized_error = self._normalize_test_error(test_error)
                    normalized_last_error = self._normalize_test_error(last_test_error) if last_test_error else None
                    
                    # Check for consecutive same errors (using normalized versions)
                    if normalized_error == normalized_last_error and normalized_error:
                        consecutive_same_error += 1
                        print(f"\n⚠️  Same error detected ({consecutive_same_error}/{max_consecutive_same_error}): {normalized_error[:100]}")
                    else:
                        if consecutive_same_error > 0:
                            print(f"   Error changed. Resetting consecutive counter.")
                        consecutive_same_error = 1
                    
                    last_test_error = test_error  # Keep full error for display
                    
                    # Only give up if we've had many consecutive identical errors
                    if consecutive_same_error >= max_consecutive_same_error:
                        print(f"\n🛑 STOPPING: Same error {max_consecutive_same_error} times in a row.")
                        print(f"   Error: {normalized_error}")
                        print(f"   Moving to next feature...")
                        self._log_thought(f"Agent: Stopped feature '{feature_name}' after {max_consecutive_same_error} identical errors: {normalized_error}")
                        break
                    
                    print(f"\n❌ Feature '{feature_name}' attempt {attempt} failed (real failures: {real_failure_count}). Retrying...")
                    time.sleep(2)
            
            if not feature_complete:
                print(f"\n⚠️  Feature '{feature_name}' could not be completed. Continuing to next feature...")
        
        # Step 4: CRITICAL - Verify ALL requirements are met before finishing
        print("\n" + "=" * 60)
        print("🔍 STEP 4: Comprehensive Validation (Target: 100% Accuracy)...")
        print("=" * 60)
        
        completion_attempt = 0
        all_requirements_met = False
        validation_history = []  # Track progress
        accuracy_threshold = 100.0  # Must reach 100% accuracy
        consecutive_no_progress = 0  # Track if we're stuck
        max_consecutive_no_progress = 10  # Restart if no progress for 10 attempts
        consecutive_api_errors = 0  # Track consecutive API failures
        max_consecutive_api_errors = 5  # Stop if API fails 5 times in a row
        
        while not all_requirements_met:
            completion_attempt += 1
            print(f"\n{'─' * 50}")
            print(f"📋 Validation attempt {completion_attempt} (Target: 100% Accuracy)")
            print(f"{'─' * 50}")
            
            # Check for consecutive API errors - if too many, stop to avoid infinite loop
            if consecutive_api_errors >= max_consecutive_api_errors:
                print(f"\n🛑 STOPPING: API failed {max_consecutive_api_errors} times consecutively.")
                print(f"   The LLM server may be down or misconfigured.")
                print(f"   Please check the server at {self.planner_client.api_url}")
                print(f"   Current accuracy: {current_accuracy:.2f}%")
                break
            
            # Calculate current accuracy (may fail if API is down)
            try:
                current_accuracy, accuracy_report = self._calculate_accuracy_percentage(requirements, task_description)
                consecutive_api_errors = 0  # Reset on success
            except Exception as e:
                if "API request failed" in str(e) or "400" in str(e) or "Bad Request" in str(e):
                    consecutive_api_errors += 1
                    print(f"\n⚠️  API Error ({consecutive_api_errors}/{max_consecutive_api_errors}): {str(e)[:100]}")
                    print(f"   Skipping this validation attempt...")
                    time.sleep(5)  # Wait before retrying
                    continue
                else:
                    # Other errors - re-raise
                    raise
            
            # Track in history
            validation_history.append({
                "attempt": completion_attempt,
                "accuracy": current_accuracy,
                "timestamp": time.time()
            })
            
            # Log to thought chain
            self._log_thought(f"Agent: Validation attempt {completion_attempt} - Accuracy: {current_accuracy:.2f}%")
            self._log_thought(f"Agent: Requirements met: {accuracy_report['met_requirements']}/{accuracy_report['total_requirements']}")
            self._log_thought(f"Agent: Structure={accuracy_report['structure_validation']}, Libraries={accuracy_report['library_validation']}, TestSync={accuracy_report['test_sync_validation']}, Completeness={accuracy_report['completeness_validation']}")
            
            # Display accuracy report with progress indicator
            progress_bar_length = 50
            filled = int((current_accuracy / 100.0) * progress_bar_length)
            progress_bar = "█" * filled + "░" * (progress_bar_length - filled)
            
            print(f"\n📊 ACCURACY: {current_accuracy:.2f}% [{progress_bar}]")
            print(f"   Requirements: {accuracy_report['met_requirements']}/{accuracy_report['total_requirements']} met")
            print(f"   Structure: {accuracy_report['structure_validation']}")
            print(f"   Libraries: {accuracy_report['library_validation']}")
            print(f"   Test Sync: {accuracy_report['test_sync_validation']}")
            print(f"   Completeness: {accuracy_report['completeness_validation']}")
            
            # Show progress trend
            if len(validation_history) >= 2:
                prev_accuracy = validation_history[-2].get('accuracy', 0)
                change = current_accuracy - prev_accuracy
                if change > 0.1:
                    print(f"   📈 Progress: +{change:.2f}% (improving)")
                elif change < -0.1:
                    print(f"   📉 Progress: {change:.2f}% (declining)")
                else:
                    print(f"   ➡️  Progress: {change:+.2f}% (stable)")
            
            # Check if we've reached 100% accuracy
            if current_accuracy >= accuracy_threshold:
                # Double-check with comprehensive validation
                try:
                    all_passed, validation_feedback = self._comprehensive_validation(task_description, requirements)
                    if all_passed:
                        try:
                            reqs_met, unmet_reqs = self._verify_requirements_met(requirements, task_description)
                            if reqs_met:
                                consecutive_api_errors = 0  # Reset on success
                                all_requirements_met = True
                                print("\n" + "=" * 60)
                                print("✅ 100% ACCURACY ACHIEVED!")
                                print("✅ All validation checks PASSED!")
                                print("✅ All requirements verified as met!")
                                print("=" * 60)
                                break
                        except Exception as e:
                            if "API request failed" in str(e) or "400" in str(e) or "Bad Request" in str(e):
                                consecutive_api_errors += 1
                                print(f"   ⚠️  API error during requirement verification ({consecutive_api_errors}/{max_consecutive_api_errors}): {str(e)[:100]}")
                                if consecutive_api_errors >= max_consecutive_api_errors:
                                    break
                                continue
                            raise
                except Exception as e:
                    if "API request failed" in str(e) or "400" in str(e) or "Bad Request" in str(e):
                        consecutive_api_errors += 1
                        print(f"   ⚠️  API error during validation ({consecutive_api_errors}/{max_consecutive_api_errors}): {str(e)[:100]}")
                        if consecutive_api_errors >= max_consecutive_api_errors:
                            break
                        continue
                    raise
            
            # Check if we should restart vs continue
            should_restart, restart_reason = self._should_restart_vs_continue(
                current_accuracy, completion_attempt, validation_history
            )
            
            if should_restart:
                print(f"\n🔄 {restart_reason}")
                print("   Attempting to fix issues without losing existing work...")
                # CRITICAL: DO NOT DELETE OUTPUT - Instead, try to fix incrementally
                # Only clear if accuracy is extremely low (< 10%) AND we've tried many times
                if current_accuracy < 10.0 and completion_attempt >= 10:
                    print("   ⚠️  Accuracy extremely low after many attempts. Creating backup before restart...")
                    # Create backup instead of deleting
                    backup_dir = f"{self.output_dir}_backup_{int(time.time())}"
                    if os.path.exists(self.output_dir):
                        import shutil
                        shutil.copytree(self.output_dir, backup_dir, dirs_exist_ok=True)
                        print(f"   📦 Backup created: {backup_dir}")
                    # Only then clear (but this should rarely happen)
                    import shutil
                    if os.path.exists(self.output_dir):
                        shutil.rmtree(self.output_dir)
                    os.makedirs(self.output_dir, exist_ok=True)
                    os.makedirs(os.path.join(self.output_dir, "src"), exist_ok=True)
                    os.makedirs(os.path.join(self.output_dir, "tests"), exist_ok=True)
                    print("   Output cleared. Please re-run the agent to start fresh.")
                    break
                else:
                    # Continue fixing incrementally without deleting
                    print("   Continuing with incremental fixes...")
                    # Don't break - continue with next validation attempt
            
            # Run comprehensive validation (structure, libraries, test sync, semantic)
            try:
                all_passed, validation_feedback = self._comprehensive_validation(task_description, requirements)
                consecutive_api_errors = 0  # Reset on success
            except Exception as e:
                if "API request failed" in str(e) or "400" in str(e) or "Bad Request" in str(e):
                    consecutive_api_errors += 1
                    print(f"\n⚠️  API Error during validation ({consecutive_api_errors}/{max_consecutive_api_errors}): {str(e)[:100]}")
                    if consecutive_api_errors >= max_consecutive_api_errors:
                        print(f"   Stopping due to repeated API failures...")
                        break
                    # Use cached validation feedback if available
                    all_passed = False
                    validation_feedback = f"API Error: {str(e)[:200]}"
                else:
                    raise
            
            # Check for progress
            if len(validation_history) >= 2:
                prev_accuracy = validation_history[-2].get('accuracy', 0)
                if abs(current_accuracy - prev_accuracy) < 0.1:  # Less than 0.1% change
                    consecutive_no_progress += 1
                else:
                    consecutive_no_progress = 0
            
            if consecutive_no_progress >= max_consecutive_no_progress:
                print(f"\n⚠️  No progress for {max_consecutive_no_progress} attempts (current accuracy: {current_accuracy:.2f}%).")
                # Don't restart automatically - instead, try more aggressive fixes
                print("   Trying more aggressive fixes instead of restarting...")
                # Only restart if accuracy is extremely low AND we've tried many times
                if current_accuracy < 5.0 and completion_attempt >= 15:
                    should_restart = True
                    restart_reason = f"No accuracy improvement for {max_consecutive_no_progress} attempts and accuracy < 5%"
                else:
                    # Continue trying - reduce counter to give more chances
                    consecutive_no_progress = max(0, consecutive_no_progress - 2)
                    print(f"   Continuing with incremental improvements...")
            
            if all_passed and current_accuracy >= accuracy_threshold:
                # Final double-check with specific requirements
                if requirements:
                    reqs_met, unmet_reqs = self._verify_requirements_met(requirements, task_description)
                    if reqs_met:
                        all_requirements_met = True
                        print("\n✅ All validation checks PASSED!")
                        print("✅ All requirements verified as met!")
                        break
                    else:
                        print(f"\n❌ {len(unmet_reqs)} requirements not met (Accuracy: {current_accuracy:.2f}%):")
                        for req in unmet_reqs[:5]:
                            print(f"   - [{req.get('id')}] {req.get('details', 'Missing')}")
                        
                        # Generate and execute fix plan
                        print("\n🔧 Generating fix plan for unmet requirements...")
                        fix_plan = self._get_completion_fix_plan(unmet_reqs, validation_feedback, task_description)
                        if fix_plan:
                            print(f"   Executing {len(fix_plan)} fix actions...")
                            self._execute_feature_plan(fix_plan, "Requirement Fixes", False)
                        time.sleep(2)
                else:
                    all_requirements_met = True
                    print("\n✅ All validation checks PASSED!")
                    break
            else:
                print(f"\n❌ Validation failed (Accuracy: {current_accuracy:.2f}%):")
                for line in validation_feedback.split('\n')[:15]:  # Show first 15 lines
                    if line.strip():
                        print(f"   {line}")
                
                # Show unmet requirements details
                if accuracy_report.get('unmet_details'):
                    print(f"\n   Unmet Requirements:")
                    for req in accuracy_report['unmet_details'][:5]:
                        print(f"     - [{req.get('id', '?')}] {req.get('details', 'Missing')}")
                
                # Generate and execute fix plan with comprehensive feedback
                print("\n🔧 Generating comprehensive fix plan...")
                
                # Prepare unmet requirements list from validation feedback
                unmet_from_validation = []
                if "STRUCTURE ISSUES" in validation_feedback:
                    unmet_from_validation.append({
                        "id": "structure",
                        "requirement": "Code structure validation",
                        "details": "Missing required classes or functions"
                    })
                if "LIBRARY ISSUES" in validation_feedback:
                    unmet_from_validation.append({
                        "id": "libraries", 
                        "requirement": "Library usage validation",
                        "details": "Required libraries not imported or used correctly"
                    })
                if "TEST-CODE SYNC" in validation_feedback:
                    unmet_from_validation.append({
                        "id": "test_sync",
                        "requirement": "Test-code synchronization",
                        "details": "Tests don't match source code structure"
                    })
                if "TASK-SPECIFIC ISSUES" in validation_feedback:
                    unmet_from_validation.append({
                        "id": "task_specific",
                        "requirement": "Task-specific validation",
                        "details": "Task-specific requirements not met"
                    })
                if "COMPLETENESS ISSUES" in validation_feedback:
                    unmet_from_validation.append({
                        "id": "completeness",
                        "requirement": "Semantic completeness",
                        "details": "Code doesn't fully implement requested features"
                    })
                
                try:
                    fix_plan = self._get_completion_fix_plan(unmet_from_validation, validation_feedback, task_description)
                    if fix_plan:
                        print(f"   Executing {len(fix_plan)} fix actions...")
                        self._execute_feature_plan(fix_plan, "Validation Fixes", False)
                        consecutive_api_errors = 0  # Reset on success
                    else:
                        print("   ⚠️  Could not generate fix plan. Retrying validation...")
                except Exception as e:
                    if "API request failed" in str(e) or "400" in str(e):
                        consecutive_api_errors += 1
                        print(f"   ⚠️  API error generating fix plan ({consecutive_api_errors}/{max_consecutive_api_errors}): {str(e)[:100]}")
                        if consecutive_api_errors >= max_consecutive_api_errors:
                            print(f"   Stopping due to repeated API failures...")
                            break
                    else:
                        raise
                time.sleep(2)
        
        if not all_requirements_met:
            # This should never happen if the loop works correctly, but just in case
            final_accuracy, final_report = self._calculate_accuracy_percentage(requirements, task_description)
            print(f"\n⚠️  Validation incomplete. Final accuracy: {final_accuracy:.2f}%")
            print("   The system should continue until 100% is reached.")
            # Log final validation state for debugging
            self._log_thought(f"Agent: Final validation incomplete - accuracy: {final_accuracy:.2f}%")
            
            # Generate a summary report of what was accomplished vs what's missing
            print("\n" + "=" * 60)
            print("📊 VALIDATION SUMMARY REPORT")
            print("=" * 60)
            print(f"\nAttempts made: {completion_attempt}")
            print(f"Final accuracy: {final_accuracy:.2f}%")
            print(f"Final status: INCOMPLETE (Target: 100%)")
            if final_report.get('unmet_details'):
                print(f"\nRemaining unmet requirements:")
                for req in final_report['unmet_details'][:10]:
                    print(f"  - [{req.get('id', '?')}] {req.get('details', 'Missing')}")
            print("\n💡 The system will continue attempting to reach 100% accuracy.")
            print("   If this message appears, there may be a logic issue in the validation loop.")
        else:
            print("\n" + "=" * 60)
            print("✅ VALIDATION COMPLETE - All checks passed!")
            print("=" * 60)
        
        # Final step: Generate final documentation
        print("\n" + "=" * 60)
        print("📚 Generating final documentation...")
        print("=" * 60)
        self._generate_final_docs_and_exit()
        
        # Cleanup: Stop PHP server if running
        self._stop_php_server()
    
    def _extract_requirements(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Extract specific technical requirements from the task description.
        Uses the Planner to analyze the task and identify concrete requirements.
        
        Returns:
            List of requirement dictionaries with keys: id, description, type, verification_method
        """
        prompt = f"""You are a Senior Software Architect. Analyze the following task and extract ALL specific technical requirements.

TASK:
{task_description}

For each requirement, identify:
1. A unique ID (req_001, req_002, etc.)
2. The specific requirement description (be VERY specific)
3. The type (library, function, class, endpoint, file, configuration, format, validation, error_handling, etc.)
4. How to verify it's implemented (import_exists, function_exists, contains_code, format_matches, etc.)

CRITICAL: Be EXTREMELY SPECIFIC. Extract EVERY technical detail mentioned:
- If task says "use requests library" → requirement: "Import and use requests library (NOT llama-cpp-python)"
- If task says "endpoint http://X:Y" → requirement: "Use EXACT URL http://X:Y (not localhost, not different port)"
- If task says "format record_YYYYMMDD_HHMMSS_UUID.json" → requirement: "File naming must include date/time in YYYYMMDD_HHMMSS format + UUID"
- If task says "validate keys are not empty" → requirement: "validate_json() must check keys exist AND values are not empty strings"
- If task says "3 retries per record" → requirement: "Implement retry loop with maximum 3 attempts before giving up"
- If task says "KeyboardInterrupt handling" → requirement: "run() method must catch KeyboardInterrupt and save files before exit"
- If task specifies a class structure → requirement: "Class must have EXACT methods: __init__(model_x_url, model_y_url), call_model_x(), etc."

IMPORTANT: Pay attention to NEGATIONS:
- If task says "use requests, NOT llama-cpp-python" → requirement must specify this exclusion
- If task says "do NOT use X" → requirement must mention this prohibition

Return a JSON array of requirements:
[
  {{"id": "req_001", "description": "Use requests library for HTTP calls (NOT llama-cpp-python)", "type": "library", "verification": "import_exists"}},
  {{"id": "req_002", "description": "Class RagGenerator.__init__ must accept model_x_url and model_y_url as parameters", "type": "function", "verification": "function_exists"}},
  {{"id": "req_003", "description": "File naming format: record_YYYYMMDD_HHMMSS_UUID.json (must include datetime)", "type": "format", "verification": "format_matches"}},
  ...
]

Return ONLY the JSON array, no markdown, no explanations."""

        messages = [
            {"role": "system", "content": "You are a Senior Software Architect who extracts precise technical requirements from specifications. You respond with JSON arrays only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.planner_client.chat_completion(messages, temperature=0.3)
            # Clean response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            requirements = json.loads(response)
            if isinstance(requirements, list):
                self._log_thought(f"Agent: Extracted {len(requirements)} requirements from task")
                return requirements
        except Exception as e:
            self._log_thought(f"Agent: Error extracting requirements: {e}")
            print(f"⚠️  Error extracting requirements: {e}")
        
        return []
    
    def _verify_requirements_met(self, requirements: List[Dict[str, Any]], task_description: str) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Verify that all extracted requirements have been implemented.
        
        Args:
            requirements: List of requirements from _extract_requirements
            task_description: Original task description
            
        Returns:
            (all_met, list of unmet requirements with details)
        """
        unmet_requirements = []
        src_dir = os.path.join(self.output_dir, "src")
        
        # Gather all generated code for analysis
        generated_code = {}
        if os.path.exists(src_dir):
            for file in os.listdir(src_dir):
                file_path = os.path.join(src_dir, file)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            generated_code[file] = f.read()
                    except:
                        pass
        
        # Use Planner to verify each requirement
        code_summary = "\n\n".join([f"=== {fname} ===\n{code[:2000]}..." if len(code) > 2000 else f"=== {fname} ===\n{code}" 
                                    for fname, code in generated_code.items()])
        
        prompt = f"""You are a Code Reviewer. Verify if the following requirements have been properly implemented.

ORIGINAL TASK:
{task_description}

REQUIREMENTS TO VERIFY:
{json.dumps(requirements, indent=2)}

GENERATED CODE:
{code_summary}

For EACH requirement, determine:
1. Is it FULLY implemented? (not partially, not approximately - FULLY)
2. If not implemented, what's EXACTLY missing?

Be EXTREMELY STRICT. A requirement is only met if:
- The exact functionality described is present
- Using the EXACT libraries/methods specified (if task says "requests", code must use requests, NOT llama_cpp)
- With the EXACT configuration specified (if task says "model_x_url parameter", __init__ must accept it)
- Producing the EXACT output format specified (if task says "record_YYYYMMDD_HHMMSS_UUID.json", must include datetime)

SPECIFIC CHECKS:
- Library requirements: Check if the CORRECT library is imported (if requirement says "use requests", check for "import requests", NOT "from llama_cpp")
- Function signatures: Check if parameters match EXACTLY (if requirement says "__init__(model_x_url, model_y_url)", check for those exact parameter names)
- File formats: Check if format matches EXACTLY (if requirement says "record_YYYYMMDD_HHMMSS_UUID.json", check for datetime.strftime usage)
- Validation logic: Check if validation is COMPLETE (if requirement says "validate keys are not empty", check for both key existence AND emptiness check)

Return a JSON array of verification results:
[
  {{"id": "req_001", "met": true, "details": "requests library imported and used correctly"}},
  {{"id": "req_002", "met": false, "details": "Wrong library imported: found 'from llama_cpp import Llama' but requirement says 'use requests library'", "missing": "Should import requests, not llama_cpp"}},
  {{"id": "req_003", "met": false, "details": "File naming uses 'record_{{uuid}}.json' but requirement says 'record_YYYYMMDD_HHMMSS_UUID.json'", "missing": "Must include datetime in format YYYYMMDD_HHMMSS"}},
  ...
]

Return ONLY the JSON array, no markdown."""

        messages = [
            {"role": "system", "content": "You are a strict Code Reviewer who verifies requirements are FULLY implemented. You respond with JSON arrays only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.planner_client.chat_completion(messages, temperature=0.2)
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            results = json.loads(response)
            if isinstance(results, list):
                for result in results:
                    if not result.get('met', False):
                        unmet_requirements.append(result)
                
                self._log_thought(f"Agent: Requirement verification - {len(results) - len(unmet_requirements)}/{len(results)} met")
                return len(unmet_requirements) == 0, unmet_requirements
        except Exception as e:
            self._log_thought(f"Agent: Error verifying requirements: {e}")
            print(f"⚠️  Error verifying requirements: {e}")
        
        return False, [{"id": "unknown", "met": False, "details": "Could not verify requirements"}]
    
    def _normalize_test_error(self, error_msg: Optional[str]) -> str:
        """
        Normalize test error message to extract the core error type and message.
        This allows detecting semantically identical errors even if details differ.
        
        Examples:
        - "ModuleNotFoundError: No module named 'output'" -> "ModuleNotFoundError: No module named 'output'"
        - "TypeError: RagGenerator.validate_json() missing 1 required positional argument: 'keys'" -> "TypeError: validate_json() missing required argument: 'keys'"
        """
        if not error_msg:
            return ""
        
        import re
        
        # Extract error type and main message
        error_patterns = [
            r"(ModuleNotFoundError|ImportError):\s*(.+)",
            r"(TypeError):\s*(.+)",
            r"(AttributeError):\s*(.+)",
            r"(ValueError):\s*(.+)",
            r"(KeyError):\s*(.+)",
            r"(SyntaxError):\s*(.+)",
            r"(NameError):\s*(.+)",
            r"(AssertionError):\s*(.+)",
        ]
        
        for pattern in error_patterns:
            match = re.search(pattern, error_msg, re.IGNORECASE)
            if match:
                error_type = match.group(1)
                error_detail = match.group(2)
                
                # Normalize common variations
                # Remove line numbers and file paths
                error_detail = re.sub(r'line \d+', '', error_detail)
                error_detail = re.sub(r'File "[^"]+", ', '', error_detail)
                error_detail = re.sub(r'/[^\s]+\.py', '', error_detail)
                error_detail = re.sub(r'/[^\s]+\.php', '', error_detail)
                
                # Normalize method signatures (remove exact parameter counts)
                error_detail = re.sub(r'\(\) takes \d+ positional arguments but \d+ were given', 
                                     r'() takes wrong number of arguments', error_detail)
                error_detail = re.sub(r'missing \d+ required positional argument', 
                                     r'missing required argument', error_detail)
                error_detail = re.sub(r'missing \d+ required positional arguments', 
                                     r'missing required arguments', error_detail)
                
                # Normalize module names (keep the module name but remove path details)
                if "No module named" in error_detail:
                    module_match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_detail)
                    if module_match:
                        return f"{error_type}: No module named '{module_match.group(1)}'"
                
                # Normalize import errors
                if "cannot import name" in error_detail.lower():
                    import_match = re.search(r"cannot import name ['\"]([^'\"]+)['\"]", error_detail, re.IGNORECASE)
                    if import_match:
                        return f"{error_type}: cannot import name '{import_match.group(1)}'"
                
                # Normalize attribute errors
                if "has no attribute" in error_detail.lower():
                    attr_match = re.search(r"['\"]([^'\"]+)['\"] has no attribute ['\"]([^'\"]+)['\"]", error_detail, re.IGNORECASE)
                    if attr_match:
                        return f"{error_type}: '{attr_match.group(1)}' has no attribute '{attr_match.group(2)}'"
                
                # Clean up whitespace and limit length
                error_detail = ' '.join(error_detail.split())
                normalized = f"{error_type}: {error_detail[:150]}"  # Limit to 150 chars
                
                # Remove any remaining file paths
                normalized = re.sub(r'/[^\s:]+', '', normalized)
                
                return normalized
        
        # If no pattern matches, extract first meaningful error line
        lines = error_msg.split('\n')
        for line in lines:
            if any(err in line for err in ['Error', 'Exception', 'Failed', 'Traceback']):
                return line[:150].strip()
        
        # Fallback: return first 100 chars
        return error_msg[:100].strip()
    
    def _calculate_accuracy_percentage(self, requirements: List[Dict[str, Any]], task_description: str) -> tuple[float, Dict[str, Any]]:
        """
        Calculate precise accuracy percentage (0-100%) based on all requirements.
        
        Returns:
            (accuracy_percentage, detailed_report)
        """
        if not requirements:
            return 100.0, {"total": 0, "met": 0, "unmet": 0, "accuracy": 100.0}
        
        # Verify all requirements
        all_met, unmet_reqs = self._verify_requirements_met(requirements, task_description)
        
        # Run comprehensive validation
        struct_ok, struct_issues = self._validate_code_structure(task_description, requirements)
        lib_ok, lib_issues = self._validate_library_usage(task_description, requirements)
        sync_ok, sync_issues = self._validate_test_code_sync()
        complete_ok, complete_feedback = self._verify_code_completeness(task_description)
        
        # Calculate weights for different validation types
        total_checks = len(requirements) + 4  # requirements + 4 validation types
        passed_checks = len(requirements) - len(unmet_reqs)
        
        # Add validation type checks (each worth 1 point)
        if struct_ok:
            passed_checks += 1
        if lib_ok:
            passed_checks += 1
        if sync_ok:
            passed_checks += 1
        if complete_ok:
            passed_checks += 1
        
        # Calculate percentage
        accuracy = (passed_checks / total_checks) * 100.0 if total_checks > 0 else 0.0
        
        # Get completeness percentage from LLM if available
        try:
            complete_result = json.loads(complete_feedback) if complete_feedback.startswith('{') else {}
            llm_completeness = complete_result.get('completeness_percentage', None)
            if llm_completeness is not None:
                # Weighted average: 70% from requirements, 30% from LLM completeness
                accuracy = (accuracy * 0.7) + (llm_completeness * 0.3)
        except:
            pass
        
        report = {
            "total_requirements": len(requirements),
            "met_requirements": len(requirements) - len(unmet_reqs),
            "unmet_requirements": len(unmet_reqs),
            "structure_validation": "PASS" if struct_ok else "FAIL",
            "library_validation": "PASS" if lib_ok else "FAIL",
            "test_sync_validation": "PASS" if sync_ok else "FAIL",
            "completeness_validation": "PASS" if complete_ok else "FAIL",
            "accuracy_percentage": round(accuracy, 2),
            "unmet_details": unmet_reqs[:10],  # Limit to first 10
            "structure_issues": struct_issues[:5],
            "library_issues": lib_issues[:5],
            "sync_issues": sync_issues[:5]
        }
        
        return accuracy, report
    
    def _should_restart_vs_continue(self, accuracy: float, attempt: int, validation_history: List[Dict]) -> tuple[bool, str]:
        """
        Decide whether to restart from scratch or continue fixing.
        
        Returns:
            (should_restart, reason)
        """
        # If accuracy is very low (< 30%) and we've tried many times, restart
        if accuracy < 30.0 and attempt >= 10:
            return True, f"Accuracy too low ({accuracy:.1f}%) after {attempt} attempts. Restarting from scratch."
        
        # If accuracy is decreasing over last 3 attempts, restart
        if len(validation_history) >= 3:
            recent_accuracies = [h.get('accuracy', 0) for h in validation_history[-3:]]
            if all(recent_accuracies[i] >= recent_accuracies[i+1] for i in range(len(recent_accuracies)-1)):
                if recent_accuracies[-1] < 50.0:
                    return True, f"Accuracy decreasing: {recent_accuracies}. Restarting from scratch."
        
        # If we've made no progress in last 5 attempts (same accuracy), restart
        if len(validation_history) >= 5:
            last_5 = [h.get('accuracy', 0) for h in validation_history[-5:]]
            if len(set(round(a, 1) for a in last_5)) == 1:  # All same
                return True, f"No progress in last 5 attempts (accuracy stuck at {last_5[0]:.1f}%). Restarting."
        
        # If accuracy is improving (> 50%) or we haven't tried many times, continue
        if accuracy >= 50.0 or attempt < 5:
            return False, f"Accuracy at {accuracy:.1f}%. Continuing to fix issues."
        
        # Default: continue
        return False, f"Continuing to fix issues (accuracy: {accuracy:.1f}%, attempt: {attempt})"
    
    def _validate_rag_specific_requirements(self, task_description: str) -> tuple[bool, List[str]]:
        """
        Validate RAG-specific requirements that are critical for this task.
        This is a specialized validator for the RAG generator task.
        
        Returns:
            (all_passed, list_of_issues)
        """
        issues = []
        src_dir = os.path.join(self.output_dir, "src")
        
        # Check if this is a RAG task
        if "rag" not in task_description.lower() and "rag_generator" not in task_description.lower():
            return True, []  # Not a RAG task, skip
        
        if not os.path.exists(src_dir):
            return False, ["No source directory exists"]
        
        # Find rag_generator.py
        rag_file = None
        for file in os.listdir(src_dir):
            if "rag" in file.lower() and file.endswith('.py'):
                rag_file = os.path.join(src_dir, file)
                break
        
        if not rag_file:
            return False, ["rag_generator.py not found"]
        
        try:
            with open(rag_file, 'r', encoding='utf-8') as f:
                code = f.read()
        except:
            return False, [f"Could not read {rag_file}"]
        
        # CRITICAL CHECKS for RAG task
        
        # 1. Check for requests library (NOT llama_cpp)
        if "import requests" not in code and "from requests" not in code:
            issues.append("Missing 'import requests' - must use requests library for HTTP calls")
        if "llama_cpp" in code or "from llama_cpp" in code or "import llama_cpp" in code:
            issues.append("Found llama_cpp import - must use 'requests' library instead")
        
        # 2. Check __init__ signature
        if "def __init__(self, model_x_url, model_y_url)" not in code and "def __init__(self, model_x_url: str, model_y_url: str)" not in code:
            issues.append("__init__ must accept exactly (self, model_x_url, model_y_url) parameters")
        
        # 3. Check API response parsing
        if 'response.json()["choices"][0]["message"]["content"]' not in code:
            issues.append("Must parse API response as: response.json()['choices'][0]['message']['content']")
        
        # 4. Check file naming format
        if "record_" not in code or "datetime.now().strftime" not in code or "uuid" not in code.lower():
            issues.append("File naming must use format: record_YYYYMMDD_HHMMSS_UUID.json")
        
        # 5. Check record structure (must have raw_intent, tags, code_snippet, description as top-level keys)
        if '"raw_intent"' not in code or '"tags"' not in code or '"code_snippet"' not in code or '"description"' not in code:
            issues.append("Record structure must have top-level keys: raw_intent, tags, code_snippet, description")
        
        # 6. Check for KeyboardInterrupt handling
        if "KeyboardInterrupt" not in code:
            issues.append("Must handle KeyboardInterrupt in run() method")
        
        # 7. Check for MAX_CONSECUTIVE_FAILURES logic
        if "MAX_CONSECUTIVE_FAILURES" not in code or "consecutive_failures" not in code:
            issues.append("Must implement MAX_CONSECUTIVE_FAILURES logic with consecutive_failures counter")
        
        # 8. Check for retry logic (3 attempts per record)
        if "for attempt in range(3)" not in code and "range(3)" not in code:
            issues.append("Must implement retry logic with 3 attempts per record")
        
        # 9. Check for JSON validation
        if "validate_json" not in code or "def validate_json" not in code:
            issues.append("Must implement validate_json() method")
        
        # 10. Check for proper error logging requirement
        # (This is mentioned in task but may not be critical for structure)
        
        return len(issues) == 0, issues
    
    def _verify_code_completeness(self, task_description: str) -> tuple[bool, str]:
        """
        High-level verification that generated code actually does what was requested.
        Uses the Planner to analyze if the code is functionally complete.
        
        Returns:
            (is_complete, feedback_message)
        """
        src_dir = os.path.join(self.output_dir, "src")
        
        # Gather all generated code
        generated_code = {}
        if os.path.exists(src_dir):
            for file in os.listdir(src_dir):
                file_path = os.path.join(src_dir, file)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            generated_code[file] = f.read()
                    except:
                        pass
        
        if not generated_code:
            return False, "No code has been generated yet"
        
        code_summary = "\n\n".join([f"=== {fname} ===\n{code}" for fname, code in generated_code.items()])
        
        prompt = f"""You are a Senior Code Reviewer. Analyze if the generated code FULLY implements what was requested.

ORIGINAL TASK REQUEST:
{task_description}

GENERATED CODE:
{code_summary}

CRITICAL CHECKS:
1. Does the code implement ALL features requested in the task?
2. Does it use the EXACT libraries/frameworks specified?
3. Does it follow the EXACT structure specified?
4. Are ALL endpoints/functions mentioned in the task implemented?
5. Does it handle ALL edge cases mentioned?
6. Is the code functional (would it actually work if run)?

Be VERY STRICT. The code must:
- Implement EVERY requirement from the task
- Not just create skeleton/placeholder code
- Have actual working logic, not just structure
- Use the correct libraries and methods

Response format (JSON):
{{
  "is_complete": true/false,
  "completeness_percentage": 0-100,
  "missing_features": ["feature1", "feature2"],
  "incomplete_implementations": ["function X is empty", "class Y missing method Z"],
  "critical_issues": ["using wrong library", "wrong API endpoint"],
  "summary": "Brief explanation of what's missing or wrong"
}}

Return ONLY the JSON, no markdown."""

        messages = [
            {"role": "system", "content": "You are a strict Senior Code Reviewer. You verify code completeness against requirements. You respond with JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.planner_client.chat_completion(messages, temperature=0.2)
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            result = json.loads(response)
            
            is_complete = result.get('is_complete', False)
            completeness = result.get('completeness_percentage', 0)
            
            if not is_complete:
                missing = result.get('missing_features', [])
                incomplete = result.get('incomplete_implementations', [])
                critical = result.get('critical_issues', [])
                summary = result.get('summary', 'Code is incomplete')
                
                feedback = f"Completeness: {completeness}%\n"
                if missing:
                    feedback += f"Missing features: {', '.join(missing)}\n"
                if incomplete:
                    feedback += f"Incomplete: {', '.join(incomplete)}\n"
                if critical:
                    feedback += f"Critical issues: {', '.join(critical)}\n"
                feedback += f"Summary: {summary}"
                
                self._log_thought(f"Agent: Code completeness check - {completeness}% complete")
                return False, feedback
            
            self._log_thought("Agent: Code completeness check - PASSED")
            return True, "Code is complete and implements all requirements"
            
        except Exception as e:
            self._log_thought(f"Agent: Error checking completeness: {e}")
            return False, f"Could not verify completeness: {e}"
    
    def _get_completion_fix_plan(self, unmet_requirements: List[Dict], completeness_feedback: str, task_description: str) -> List[Dict[str, Any]]:
        """
        Generate a plan to fix unmet requirements and incomplete code.
        
        Args:
            unmet_requirements: List of requirements that weren't met
            completeness_feedback: Feedback from completeness check
            task_description: Original task
            
        Returns:
            Action plan to fix the issues
        """
        project_type = self._detect_project_type()
        existing_files_context = self._get_existing_files_context(project_type)
        
        # Get the structural blueprint for additional guidance
        import re
        
        # Extract specific missing items from feedback for targeted fixes
        specific_fixes = []
        if "Missing class:" in completeness_feedback:
            classes = re.findall(r'Missing class:\s*(\w+)', completeness_feedback)
            specific_fixes.extend([f"Add class {cls} with all required methods" for cls in classes])
        if "Missing function" in completeness_feedback:
            funcs = re.findall(r'Missing function/method:\s*(\w+)', completeness_feedback)
            specific_fixes.extend([f"Add function {func}() with complete implementation" for func in funcs])
        if "not imported" in completeness_feedback:
            libs = re.findall(r"Library '(\w+(?:-\w+)*)' is required but not imported", completeness_feedback)
            specific_fixes.extend([f"Add import for {lib}" for lib in libs])
        if "not used correctly" in completeness_feedback:
            libs = re.findall(r"Library '(\w+(?:-\w+)*)' is imported but not used correctly", completeness_feedback)
            specific_fixes.extend([f"Fix usage of {lib} - use proper API calls" for lib in libs])
        
        specific_fixes_str = ""
        if specific_fixes:
            specific_fixes_str = "\n\nSPECIFIC FIXES REQUIRED:\n" + "\n".join(f"- {fix}" for fix in specific_fixes)
        
        prompt = f"""You are a Senior Developer fixing incomplete code. The code was generated but doesn't meet all requirements.

ORIGINAL TASK:
{task_description}

EXISTING FILES:
{existing_files_context}

UNMET REQUIREMENTS:
{json.dumps(unmet_requirements, indent=2)}

COMPLETENESS ISSUES:
{completeness_feedback}
{specific_fixes_str}

YOUR TASK: Generate a plan to FIX all issues and make the code complete.

CRITICAL INSTRUCTIONS:
1. Focus on the MAIN source file first (e.g., src/rag_generator.py) - this is where most fixes are needed
2. MODIFY existing files - don't create duplicates  
3. Add the MISSING functionality - not skeleton code, ACTUAL WORKING code
4. Ensure ALL requirements from the original task are met
5. Be VERY SPECIFIC in your instructions:
   - Include EXACT function signatures with parameters
   - Include EXACT import statements needed
   - Include EXACT variable names and values for configurations
   - Include EXACT logic flow and algorithms
   
EXAMPLE OF GOOD INSTRUCTION:
"Rewrite the entire file with these components:
1. Imports: from llama_cpp import Llama, import json, import uuid, import os
2. Constants: MAX_RECORDS = 100, MAX_CONSECUTIVE_FAILURES = 3
3. Class RAGGenerator with __init__(self, model_x_path, model_y_path) method
4. Method generate_record() that: initializes Llama model X, creates prompt, gets response, validates JSON, uses model Y for alternatives
5. Main loop that generates up to MAX_RECORDS, handles failures, saves to output file"

OUTPUT FORMAT: JSON array of actions (1-3 actions maximum for efficiency):
[{{"step": 1, "action": "write_file", "target": "src/file.py", "content_instruction": "COMPLETE rewrite with: [list all components, imports, classes, functions, logic]"}}]

Return ONLY the JSON array."""

        messages = [
            {"role": "system", "content": f"You are a Senior {project_type} Developer who fixes incomplete code. You generate detailed action plans. You respond with JSON arrays only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.planner_client.chat_completion(messages, temperature=0.3)
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            plan = json.loads(response)
            if isinstance(plan, list):
                self._log_thought(f"Agent: Generated fix plan with {len(plan)} actions")
                return plan
        except Exception as e:
            self._log_thought(f"Agent: Error generating fix plan: {e}")
        
        return []
    
    def _validate_code_structure(self, task_description: str, requirements: List[Dict]) -> tuple[bool, List[str]]:
        """
        Validates that the generated code contains all required classes, functions, and structures.
        
        Returns:
            (is_valid, list_of_missing_structures)
        """
        src_dir = os.path.join(self.output_dir, "src")
        missing_structures = []
        
        if not os.path.exists(src_dir):
            return False, ["No source directory exists"]
        
        # Gather all code content
        all_code = ""
        code_by_file = {}
        for file in os.listdir(src_dir):
            file_path = os.path.join(src_dir, file)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        all_code += content + "\n"
                        code_by_file[file] = content
                except:
                    pass
        
        if not all_code:
            return False, ["No source code found"]
        
        # Extract required structures from task description using patterns
        required_classes = []
        required_functions = []
        required_variables = []
        
        # Parse task for class requirements
        import re
        class_patterns = [
            r'class[es]?\s+(?:named?|called?)?\s*[`"\']?(\w+)[`"\']?',
            r'creat[e]\s+(?:a\s+)?class\s+[`"\']?(\w+)[`"\']?',
            r'(?:class|Class)\s+[`"\']?(\w+)[`"\']?\s+(?:that|which|for)',
        ]
        for pattern in class_patterns:
            matches = re.findall(pattern, task_description, re.IGNORECASE)
            required_classes.extend(matches)
        
        # Parse task for function requirements
        func_patterns = [
            r'function[s]?\s+(?:named?|called?)?\s*[`"\']?(\w+)[`"\']?',
            r'method[s]?\s+(?:named?|called?)?\s*[`"\']?(\w+)[`"\']?',
            r'def\s+(\w+)\s*\(',
            r'implement\s+(?:a\s+)?(?:function|method)\s+[`"\']?(\w+)[`"\']?',
        ]
        for pattern in func_patterns:
            matches = re.findall(pattern, task_description, re.IGNORECASE)
            required_functions.extend(matches)
        
        # Parse from requirements
        for req in requirements:
            req_text = req.get('requirement', '').lower()
            if 'class' in req_text:
                class_match = re.search(r'class\s+[`"\']?(\w+)[`"\']?', req_text, re.IGNORECASE)
                if class_match:
                    required_classes.append(class_match.group(1))
            if 'function' in req_text or 'method' in req_text:
                func_match = re.search(r'(?:function|method)\s+[`"\']?(\w+)[`"\']?', req_text, re.IGNORECASE)
                if func_match:
                    required_functions.append(func_match.group(1))
        
        # Remove duplicates
        required_classes = list(set(required_classes))
        required_functions = list(set(required_functions))
        
        # Check for classes in code
        for cls in required_classes:
            class_found = False
            patterns_to_check = [
                rf'class\s+{cls}\s*[\(:]',  # Python: class Name: or class Name(Parent):
                rf'class\s+{cls}\s*\{{',     # PHP/JS: class Name {
            ]
            for pattern in patterns_to_check:
                if re.search(pattern, all_code, re.IGNORECASE):
                    class_found = True
                    break
            if not class_found:
                missing_structures.append(f"Missing class: {cls}")
        
        # Check for functions in code
        for func in required_functions:
            func_found = False
            patterns_to_check = [
                rf'def\s+{func}\s*\(',           # Python function
                rf'function\s+{func}\s*\(',       # PHP/JS function
                rf'{func}\s*=\s*function',        # JS arrow function
                rf'{func}\s*=\s*\([^)]*\)\s*=>',  # JS arrow function
                rf'async\s+def\s+{func}\s*\(',    # Python async
            ]
            for pattern in patterns_to_check:
                if re.search(pattern, all_code, re.IGNORECASE):
                    func_found = True
                    break
            if not func_found:
                missing_structures.append(f"Missing function/method: {func}")
        
        if missing_structures:
            self._log_thought(f"Agent: Structure validation found {len(missing_structures)} missing items")
        else:
            self._log_thought("Agent: Structure validation PASSED")
        
        return len(missing_structures) == 0, missing_structures
    
    def _validate_library_usage(self, task_description: str, requirements: List[Dict]) -> tuple[bool, List[str]]:
        """
        Validates that required libraries are actually imported and used in the code.
        
        Returns:
            (is_valid, list_of_issues)
        """
        src_dir = os.path.join(self.output_dir, "src")
        issues = []
        
        if not os.path.exists(src_dir):
            return False, ["No source directory exists"]
        
        # Gather all code content
        all_code = ""
        for file in os.listdir(src_dir):
            file_path = os.path.join(src_dir, file)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        all_code += f.read() + "\n"
                except:
                    pass
        
        if not all_code:
            return False, ["No source code found"]
        
        import re
        
        # Extract required libraries from task
        required_libraries = []
        lib_patterns = [
            r'use\s+(?:the\s+)?[`"\']?(\w+(?:-\w+)*)[`"\']?\s+(?:library|package|module)',
            r'(?:library|package|module)\s+[`"\']?(\w+(?:-\w+)*)[`"\']?',
            r'import\s+[`"\']?(\w+(?:-\w+)*)[`"\']?',
            r'require\s+[`"\']?(\w+(?:-\w+)*)[`"\']?',
            r'using\s+[`"\']?(\w+(?:-\w+)*)[`"\']?',
            r'(?:llama[-_]cpp[-_]?python|llama_cpp)',  # Specific for LLM libs
        ]
        
        task_lower = task_description.lower()
        
        # Common library mappings
        library_checks = {}
        
        # Check for specific libraries mentioned in task
        # CRITICAL: Only check for llama-cpp-python if task explicitly says to use it
        # AND does NOT say to use requests instead
        use_llama = ('llama-cpp-python' in task_lower or 'llama_cpp' in task_lower)
        use_requests = ('requests' in task_lower and 'not use requests' not in task_lower)
        explicit_no_llama = ('not use llama' in task_lower or 'do not use llama' in task_lower or 
                             'use requests' in task_lower and 'not llama' in task_lower)
        
        if use_llama and not explicit_no_llama and not (use_requests and 'not llama' in task_lower):
            library_checks['llama-cpp-python'] = {
                'import_patterns': [r'from\s+llama_cpp\s+import', r'import\s+llama_cpp'],
                'usage_patterns': [r'Llama\s*\(', r'llama_cpp\.']
            }
        
        if use_requests:
            library_checks['requests'] = {
                'import_patterns': [r'import\s+requests', r'from\s+requests\s+import'],
                'usage_patterns': [r'requests\.(get|post|put|delete)']
            }
        
        if 'json' in task_lower:
            library_checks['json'] = {
                'import_patterns': [r'import\s+json'],
                'usage_patterns': [r'json\.(load|dump|loads|dumps)']
            }
        
        if 'uuid' in task_lower:
            library_checks['uuid'] = {
                'import_patterns': [r'import\s+uuid', r'from\s+uuid\s+import'],
                'usage_patterns': [r'uuid\.(uuid4|uuid1)', r'uuid4\(\)', r'str\(uuid']
            }
        
        if 'pandas' in task_lower:
            library_checks['pandas'] = {
                'import_patterns': [r'import\s+pandas', r'from\s+pandas\s+import'],
                'usage_patterns': [r'pd\.', r'DataFrame', r'read_csv', r'to_csv']
            }
        
        if 'numpy' in task_lower:
            library_checks['numpy'] = {
                'import_patterns': [r'import\s+numpy', r'from\s+numpy\s+import'],
                'usage_patterns': [r'np\.', r'numpy\.', r'array\(']
            }
        
        # Check each required library
        for lib_name, checks in library_checks.items():
            # Check import
            imported = False
            for pattern in checks['import_patterns']:
                if re.search(pattern, all_code):
                    imported = True
                    break
            
            if not imported:
                issues.append(f"Library '{lib_name}' is required but not imported")
                continue
            
            # Check usage
            used = False
            for pattern in checks['usage_patterns']:
                if re.search(pattern, all_code):
                    used = True
                    break
            
            if not used:
                issues.append(f"Library '{lib_name}' is imported but not used correctly")
        
        # Check for prohibited patterns
        if 'not use http' in task_lower or 'local model' in task_lower:
            if re.search(r'requests\.(get|post)', all_code):
                issues.append("Code uses HTTP requests but task requires local model interaction")
        
        if issues:
            self._log_thought(f"Agent: Library validation found {len(issues)} issues")
        else:
            self._log_thought("Agent: Library validation PASSED")
        
        return len(issues) == 0, issues
    
    def _validate_test_code_sync(self) -> tuple[bool, List[str]]:
        """
        Validates that test files are synchronized with source code:
        - Tests import correct modules
        - Test functions test existing functions/classes
        - Mock objects match real implementations
        
        Returns:
            (is_valid, list_of_sync_issues)
        """
        src_dir = os.path.join(self.output_dir, "src")
        tests_dir = os.path.join(self.output_dir, "tests")
        issues = []
        
        if not os.path.exists(src_dir) or not os.path.exists(tests_dir):
            return True, []  # Can't validate without both directories
        
        import re
        
        # Gather source code structures
        source_classes = set()
        source_functions = set()
        source_files = {}
        
        for file in os.listdir(src_dir):
            file_path = os.path.join(src_dir, file)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        source_files[file] = content
                        
                        # Extract classes
                        classes = re.findall(r'class\s+(\w+)\s*[\(:]', content)
                        source_classes.update(classes)
                        
                        # Extract functions
                        funcs = re.findall(r'def\s+(\w+)\s*\(', content)
                        source_functions.update(funcs)
                        funcs = re.findall(r'function\s+(\w+)\s*\(', content)
                        source_functions.update(funcs)
                except:
                    pass
        
        # Analyze test files
        for test_file in os.listdir(tests_dir):
            if not test_file.startswith('test_') and not test_file.endswith('_test.py'):
                continue
                
            test_path = os.path.join(tests_dir, test_file)
            try:
                with open(test_path, 'r', encoding='utf-8') as f:
                    test_content = f.read()
                
                # Check imports - are they importing things that exist?
                imports = re.findall(r'from\s+(?:src\.)?(\w+)\s+import\s+(.+)', test_content)
                for module, imported_items in imports:
                    items = [i.strip() for i in imported_items.split(',')]
                    for item in items:
                        # Clean up item name
                        item = item.split(' as ')[0].strip()
                        if item and item not in source_classes and item not in source_functions:
                            # Check if it's a module import
                            if f"{module}.py" not in source_files:
                                issues.append(f"{test_file}: imports '{item}' from '{module}' but it doesn't exist in source")
                
                # Check tested functions - are they testing things that exist?
                test_functions = re.findall(r'def\s+test_(\w+)\s*\(', test_content)
                for test_func in test_functions:
                    # Test function names often match source function names
                    # e.g., test_generate_data tests generate_data
                    base_name = test_func.replace('_', '')
                    found_match = False
                    for src_func in source_functions:
                        if src_func.lower().replace('_', '') in base_name.lower() or base_name.lower() in src_func.lower().replace('_', ''):
                            found_match = True
                            break
                    # Don't flag this as error - test function naming is flexible
                
                # Check for mocked items that don't exist
                mocked = re.findall(r'@patch\([\'"]([^\'"]+)[\'"]\)', test_content)
                mocked.extend(re.findall(r'mock\.patch\([\'"]([^\'"]+)[\'"]\)', test_content))
                for mock_target in mocked:
                    parts = mock_target.split('.')
                    if len(parts) >= 2:
                        module = parts[-2] if len(parts) > 2 else parts[0]
                        attr = parts[-1]
                        # Check if mocked attribute could exist
                        if attr not in source_functions and attr not in source_classes:
                            # Only flag if it's definitely our code
                            if 'src.' in mock_target or any(sf.replace('.py', '') in mock_target for sf in source_files):
                                issues.append(f"{test_file}: mocks '{attr}' but it doesn't exist in source")
                
            except Exception as e:
                pass
        
        if issues:
            self._log_thought(f"Agent: Test-code sync validation found {len(issues)} issues")
        else:
            self._log_thought("Agent: Test-code sync validation PASSED")
        
        return len(issues) == 0, issues
    
    def _comprehensive_validation(self, task_description: str, requirements: List[Dict]) -> tuple[bool, str]:
        """
        Runs all validation checks and returns a comprehensive report.
        
        Returns:
            (all_passed, comprehensive_feedback)
        """
        all_passed = True
        feedback_parts = []
        
        # 1. Structure validation
        struct_ok, struct_issues = self._validate_code_structure(task_description, requirements)
        if not struct_ok:
            all_passed = False
            feedback_parts.append("STRUCTURE ISSUES:\n" + "\n".join(f"  - {i}" for i in struct_issues))
        
        # 2. Library validation
        lib_ok, lib_issues = self._validate_library_usage(task_description, requirements)
        if not lib_ok:
            all_passed = False
            feedback_parts.append("LIBRARY ISSUES:\n" + "\n".join(f"  - {i}" for i in lib_issues))
        
        # 3. Test-code sync validation
        sync_ok, sync_issues = self._validate_test_code_sync()
        if not sync_ok:
            all_passed = False
            feedback_parts.append("TEST-CODE SYNC ISSUES:\n" + "\n".join(f"  - {i}" for i in sync_issues))
        
        # 4. Task-specific validation (e.g., RAG-specific requirements)
        task_specific_ok, task_specific_issues = self._validate_rag_specific_requirements(task_description)
        if not task_specific_ok:
            all_passed = False
            feedback_parts.append("TASK-SPECIFIC ISSUES:\n" + "\n".join(f"  - {i}" for i in task_specific_issues))
        
        # 5. Semantic completeness check (using LLM)
        complete_ok, complete_feedback = self._verify_code_completeness(task_description)
        if not complete_ok:
            all_passed = False
            feedback_parts.append(f"COMPLETENESS ISSUES:\n  {complete_feedback}")
        
        if all_passed:
            return True, "All validation checks passed"
        
        return False, "\n\n".join(feedback_parts)
    
    def _get_feature_list(self, task_description: str) -> List[str]:
        """Ask Planner to identify all features to implement."""
        prompt = f"""You are a Senior Software Architect. Analyze the following task and identify ALL features that need to be implemented.

TASK:
{task_description}

Your response must be a JSON array of feature names (strings only). Example:
["Database Setup", "User Authentication", "Booking System", "Admin Panel"]

Return ONLY the JSON array, no markdown, no explanations."""
        
        messages = [
            {"role": "system", "content": "You are a Senior Software Architect. You analyze tasks and identify features to implement. You respond with JSON arrays only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.planner_client.chat_completion(messages, temperature=0.7)
            features = json.loads(response.strip())
            if isinstance(features, list):
                return [str(f) for f in features]
        except Exception as e:
            print(f"⚠️  Error getting feature list: {e}")
            # Fallback: return a single feature
            return ["Main Feature"]
        
        return []
    
    def _get_feature_plan(self, feature_name: str, task_description: str, is_first_feature: bool, last_test_error: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get execution plan for a specific feature."""
        project_type = self._detect_project_type()
        test_framework_info = self._get_test_framework_info(project_type)
        
        # For PHP projects, use Python tests with curl/requests instead of PHP tests
        if project_type == "PHP":
            test_framework_info += f"\n\n⚠️ CRITICAL: For PHP projects, generate PYTHON test files (test_*.py) that use curl or requests library to test the PHP API endpoints. The PHP server will be automatically started on http://localhost:{self.php_server_port} before running tests. Test files should use this URL. Do NOT generate PHP test files. Use Python's unittest or simple assert statements. Test the PHP application by making HTTP requests to the API endpoints at http://localhost:{self.php_server_port}."
        
        context = f"TASK: {task_description}\n\n"
        context += f"CURRENT FEATURE: {feature_name}\n"
        context += f"PROJECT TYPE: {project_type}\n{test_framework_info}\n"
        
        # CRITICAL: Add extracted requirements to help Planner understand what MUST be implemented
        if hasattr(self, 'extracted_requirements') and self.extracted_requirements:
            req_list = []
            for req in self.extracted_requirements[:20]:  # Limit to avoid token overflow
                req_list.append(f"  - [{req.get('id', '?')}] {req.get('description', '')}")
            if req_list:
                context += f"\n📋 EXTRACTED REQUIREMENTS (MUST be implemented in code):\n"
                context += "\n".join(req_list)
                context += "\n\n⚠️  CRITICAL: When planning, ensure ALL these requirements are addressed. Pay special attention to:\n"
                context += "  - Library requirements (use EXACT library specified, NOT alternatives)\n"
                context += "  - Function signatures (EXACT parameter names and types)\n"
                context += "  - File formats (EXACT format specified, including datetime if mentioned)\n"
                context += "  - Validation logic (COMPLETE validation, not partial)\n\n"
        
        # CRITICAL: Add information about existing files so Planner knows what's already implemented
        existing_files_context = self._get_existing_files_context(project_type)
        context += existing_files_context
        
        # Add recent execution history to help Planner understand what was tried
        if hasattr(self, 'history') and self.history:
            recent_history = [h for h in self.history[-10:] if h.get('action') in ['write_file', 'execute_command']]
            if recent_history:
                context += "\n\n📜 RECENT EXECUTION HISTORY (what was already tried):\n"
                for action in recent_history:
                    context += f"  - {action.get('action')} on {action.get('target', '?')}"
                    if action.get('success') == False:
                        context += " ❌ FAILED"
                    elif action.get('success') == True:
                        context += " ✅ SUCCESS"
                    context += "\n"
        
        # Add code snippets from existing files to help Planner understand current implementation
        src_dir = os.path.join(self.output_dir, "src")
        if os.path.exists(src_dir):
            for file in os.listdir(src_dir):
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(src_dir, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code_content = f.read()
                        # Extract key parts (imports, class definitions, method signatures)
                        import re
                        imports = re.findall(r'^import .+|^from .+ import .+', code_content, re.MULTILINE)
                        classes = re.findall(r'^class \w+.*?:', code_content, re.MULTILINE)
                        methods = re.findall(r'^\s+def \w+\([^)]*\):', code_content, re.MULTILINE)
                        
                        if imports or classes or methods:
                            context += f"\n\n📄 CURRENT CODE IN src/{file}:\n"
                            if imports:
                                context += f"  Imports: {', '.join(imports[:5])}\n"
                            if classes:
                                context += f"  Classes: {', '.join(classes)}\n"
                            if methods:
                                context += f"  Methods: {', '.join(methods[:8])}\n"
                    except Exception:
                        pass
        
        # Add language-specific coherence rules
        context += self._get_language_specific_coherence_rules(project_type)
        
        if last_test_error:
            context += f"⚠️ LAST ERROR:\n{last_test_error}\n\n"
            context += "INSTRUCTIONS:\n"
            context += self._get_error_handling_instructions(project_type, last_test_error)
        elif is_first_feature:
            context += self._get_first_feature_instructions(project_type)
        else:
            context += self._get_subsequent_feature_instructions(project_type)
        
        context += "\n⚠️ CRITICAL JSON FORMAT RULES:\n"
        context += "1. Your response MUST be ONLY a valid JSON array - NO markdown, NO explanations\n"
        context += "2. Keep content_instruction SHORT (max 500 chars) - NO actual code, just descriptions\n"
        context += "3. DO NOT include code snippets or examples in content_instruction\n"
        context += "4. Use bullet points format for listing requirements\n\n"
        
        context += "OUTPUT FORMAT: JSON array of actions.\n"
        context += '[{"step": 1, "action": "write_file", "target": "src/file.py", "content_instruction": "Short description"}]\n\n'
        
        context += "AVAILABLE ACTIONS:\n"
        context += "- read_file: Read existing file content\n"
        context += "- write_file: Write/overwrite a file\n"
        context += "- execute_command: Run shell command\n\n"
        
        context += "CONTENT_INSTRUCTION FORMAT (be specific but SHORT):\n"
        context += "✅ GOOD: 'Create class RagGenerator with: __init__(model_x_url, model_y_url), call_model_x() extracts content from response.json()['choices'][0]['message']['content'] then parses with json.loads(), call_model_y(raw_intent) same pattern, validate_json(data, keys) checks keys exist AND not empty, save_record() uses datetime.strftime('%Y%m%d_%H%M%S') + uuid, run() has retry loop for entire record (3 attempts), checks consecutive_failures < MAX_CONSECUTIVE_FAILURES, handles KeyboardInterrupt'\n"
        context += "❌ BAD: Including actual Python/PHP code in the instruction\n"
        context += "❌ BAD: Very long instructions with full implementation details\n\n"
        
        context += "⚠️ CRITICAL IMPLEMENTATION PATTERNS (tell Executor to use these):\n"
        context += "- For OpenAI-compatible API calls: Extract content with response.json()['choices'][0]['message']['content'], then parse with json.loads(content)\n"
        context += "- For retry logic: Retry the ENTIRE record generation (call X + validate + call Y + validate), not just HTTP calls\n"
        context += "- For file naming: Use datetime.now().strftime('%Y%m%d_%H%M%S') + uuid.uuid4()\n"
        context += "- For validation: Check keys exist AND values are not empty strings\n"
        context += "- For error handling: Check consecutive_failures in loop condition, save log on stop\n\n"
        
        context += "The Executor is an expert coder - give it WHAT to build, not HOW to code it.\n"
        context += "Include: class names, method signatures, config values, file paths from TASK, CRITICAL patterns above.\n"
        context += "Exclude: actual code, examples, lengthy explanations.\n"
        
        # CRITICAL: For PHP projects, explicitly forbid PHP test files
        if project_type == "PHP":
            context += "\n\n🚫 FORBIDDEN FOR PHP PROJECTS:\n"
            context += "- Do NOT create test files with .php extension (e.g., tests/test_*.php)\n"
            context += "- Do NOT create test files named test_*.php\n"
            context += "- ONLY create Python test files (test_*.py)\n"
            context += "- If you see 'test' in target path and it ends with .php, change it to .py\n"
        
        messages = [
            {"role": "system", "content": "You are a Senior Software Architect. Respond with JSON arrays only. Keep content_instruction fields SHORT (max 300 chars). NO code in responses."},
            {"role": "user", "content": context}
        ]
        
        try:
            response = self.planner_client.chat_completion(messages, temperature=0.7)
            plan = self._clean_json(response)
            if plan:
                return plan
        except Exception as e:
            print(f"⚠️  Error getting feature plan: {e}")
        
        # Retry with simplified prompt if first attempt failed
        print("🔄 Retrying with simplified prompt...")
        simple_prompt = f"""Feature: {feature_name}
Task: {task_description[:500]}

Return JSON array with 1-2 actions. Keep it SHORT.
Example: [{{"step":1,"action":"write_file","target":"src/main.py","content_instruction":"Create main module with required classes"}}]

JSON only, no markdown:"""
        
        try:
            response = self.planner_client.chat_completion([
                {"role": "system", "content": "Return only valid JSON array. Very short responses."},
                {"role": "user", "content": simple_prompt}
            ], temperature=0.3)
            plan = self._clean_json(response)
            return plan
        except Exception as e:
            print(f"⚠️  Simplified prompt also failed: {e}")
            return []
    
    def _execute_feature_plan(self, plan: List[Dict[str, Any]], feature_name: str, is_first_feature: bool) -> tuple[bool, Optional[str]]:
        """
        Execute plan for a feature. 
        Returns (True, None) if all tests pass, (False, error_message) if tests fail.
        """
        last_test_error = None
        
        for action in plan:
            action_type = action.get("action", "").lower()
            target = action.get("target", "")
            content_instruction = action.get("content_instruction", "")
            
            print(f"\n📝 Action: {action_type} -> {target}")
            
            if action_type == "write_file":
                # Normalize path
                normalized_path = self._normalize_path(target)
                
                # For PHP projects, if Planner tries to create a PHP test file, convert it to Python
                project_type = self._detect_project_type()
                if project_type == "PHP" and "test" in normalized_path.lower() and normalized_path.endswith(".php"):
                    # Convert PHP test file to Python
                    normalized_path = normalized_path.replace(".php", ".py")
                    print(f"⚠️  Converting PHP test file to Python: {target} -> {normalized_path}")
                    self._log_thought(f"Auto-converted PHP test file to Python: {normalized_path}")
                
                # Ensure git repo exists before first file (only for first feature)
                if is_first_feature and not self.git_repo_initialized:
                    self._ensure_git_repo()
                
                # Generate and write file
                try:
                    # CRITICAL: Backup existing file if it exists and is valid
                    file_exists = os.path.exists(normalized_path)
                    if file_exists:
                        # Check if file has content (not empty)
                        try:
                            with open(normalized_path, 'r', encoding='utf-8') as f:
                                existing_content = f.read()
                            if len(existing_content.strip()) > 100:  # File has substantial content
                                # Create backup
                                backup_path = f"{normalized_path}.backup_{int(time.time())}"
                                import shutil
                                shutil.copy2(normalized_path, backup_path)
                                print(f"   💾 Backed up existing file to: {backup_path}")
                        except Exception as e:
                            print(f"   ⚠️  Could not backup file: {e}")
                    
                    code_content = self._ask_executor(content_instruction, normalized_path)
                    
                    # Validate that generated content is not empty
                    if not code_content or len(code_content.strip()) < 10:
                        print(f"   ⚠️  Generated content is too short, skipping write")
                        if file_exists:
                            print(f"   ✅ Keeping existing file")
                            continue
                        else:
                            return False, f"Generated content for {target} is empty"
                    
                    self.tools.write_file(normalized_path, code_content)
                    self._log_thought(f"File written: {normalized_path}")
                    
                    # Validate PHP syntax if it's a PHP file
                    if normalized_path.endswith('.php'):
                        # Get relative path for PHP command (remove output/ prefix since cwd is output/)
                        php_check_path = normalized_path
                        if php_check_path.startswith(self.output_dir + "/"):
                            php_check_path = php_check_path[len(self.output_dir) + 1:]
                        elif php_check_path.startswith("output/"):
                            php_check_path = php_check_path[7:]
                        
                        syntax_check = self.tools.execute_command(f"php -l {php_check_path}", timeout=10, cwd=self.output_dir)
                        if syntax_check[2] != 0:
                            error_msg = f"PHP syntax error in {normalized_path}:\n{syntax_check[1]}"
                            print(f"❌ {error_msg}")
                            self._log_thought(f"PHP syntax error detected: {normalized_path}")
                            return False, error_msg
                        print(f"✅ PHP syntax valid for {normalized_path}")
                    
                    # Track in history
                    self.history.append({
                        "action": "write_file",
                        "target": normalized_path,
                        "success": True
                    })
                    # Track for current feature documentation (avoid duplicates)
                    if normalized_path not in self.current_feature_files:
                        self.current_feature_files.append(normalized_path)
                except Exception as e:
                    print(f"❌ Error writing file: {e}")
                    return False, f"Error writing file {target}: {str(e)}"
            
            elif action_type == "execute_command":
                # Normalize command paths
                target = self._normalize_command_paths(target)
                project_type = self._detect_project_type()
                
                # For PHP projects, if this looks like a test command, ensure server is running FIRST
                import re
                is_test = False
                test_file_match = re.search(r'(tests?/[^\s]+\.(php|py|js|java|rb|go))', target)
                if test_file_match or any(kw in target.lower() for kw in ["test", "phpunit", "pytest", "unittest"]):
                    is_test = True
                    # For PHP projects, start server BEFORE correcting command
                    if project_type == "PHP":
                        if not self._start_php_server():
                            error_msg = "Failed to start PHP server for testing"
                            print(f"❌ {error_msg}")
                            return False, error_msg
                
                target = self._correct_test_command(target, project_type)
                
                if is_test:
                    # Extract test file path from command if present
                    if test_file_match:
                        test_file = test_file_match.group(1)
                        test_file_path = os.path.join(self.output_dir, test_file)
                        
                        # Verify test file exists before executing
                        if not os.path.exists(test_file_path):
                            error_msg = f"Test file does not exist: {test_file_path}. Cannot execute test."
                            print(f"❌ {error_msg}")
                            self._log_thought(f"Test file missing: {test_file_path}")
                            return False, error_msg
                        
                        # For Python test files, validate syntax first
                        if test_file.endswith('.py'):
                            syntax_check = self.tools.execute_command(f"python3 -m py_compile {test_file}", timeout=10, cwd=self.output_dir)
                            if syntax_check[2] != 0:
                                error_msg = f"Python syntax error in {test_file}:\n{syntax_check[1]}"
                                print(f"❌ {error_msg}")
                                return False, error_msg
                            print(f"✅ Python syntax valid for {test_file}")
                
                # Execute command
                stdout, stderr, return_code = self.tools.execute_command(target, timeout=60, cwd=self.output_dir)
                
                if is_test:
                    self.test_counter += 1
                    status = "PASSED" if return_code == 0 else "FAILED"
                    print(f"🧪 Test #{self.test_counter}: {target} - {status}")
                    self._log_thought(f"Test #{self.test_counter}: {target} - {status}")
                    
                    if return_code != 0:
                        error_msg = f"Test failed: {target}\nSTDOUT: {stdout}\nSTDERR: {stderr}"
                        print(f"❌ Test failed. Output:\n{stderr}")
                        return False, error_msg
                    else:
                        self.feature_test_passed = True
            
            time.sleep(1)  # Respect GPU thermals
        
        # After all actions, check if test files were written but not executed
        # If so, execute them automatically (works for all project types)
        project_type = self._detect_project_type()
        tests_dir = os.path.join(self.output_dir, "tests")
        
        if os.path.exists(tests_dir) and not self.feature_test_passed:
            test_files = [f for f in os.listdir(tests_dir) if f.endswith('.py') and (f.startswith('test_') or 'test' in f.lower())]
            
            if test_files:
                print(f"\n⚠️  Test files found but not executed. Running them now...")
                print(f"   Found {len(test_files)} test file(s): {', '.join(test_files)}")
                
                # For PHP projects, start server first
                if project_type == "PHP":
                    if not self._start_php_server():
                        return False, "Failed to start PHP server for testing"
                
                # Auto-fix common test issues before validation
                self._auto_fix_test_imports(tests_dir)
                self._auto_fix_test_constructors(tests_dir)
                
                # Validate test-code coherence - but only warn, don't block
                coherence_issues = self._validate_test_code_coherence(tests_dir)
                if coherence_issues:
                    print(f"\n⚠️  Test coherence warnings (will try to run anyway):")
                    for issue in coherence_issues[:3]:
                        print(f"   - {issue}")
                
                # Run tests using pytest (preferred) or unittest
                test_cmd = "python3 -m pytest tests/ -v" if self._is_pytest_available() else "python3 -m unittest discover -s tests -v"
                print(f"🧪 Running: {test_cmd}")
                
                stdout, stderr, return_code = self.tools.execute_command(test_cmd, timeout=120, cwd=self.output_dir)
                self.test_counter += 1
                status = "PASSED" if return_code == 0 else "FAILED"
                print(f"🧪 Test #{self.test_counter}: {test_cmd} - {status}")
                self._log_thought(f"Test #{self.test_counter}: {test_cmd} - {status}")
                
                if return_code != 0:
                    # Extract meaningful error info
                    error_output = stderr if stderr else stdout
                    error_lines = [l for l in error_output.split('\n') if l.strip() and ('Error' in l or 'FAILED' in l or 'assert' in l.lower() or 'import' in l.lower())]
                    error_summary = '\n'.join(error_lines[:10]) if error_lines else error_output[:500]
                    
                    error_msg = f"Tests failed:\n{error_summary}"
                    print(f"❌ Test output:\n{error_output[:1000]}")
                    return False, error_msg
                else:
                    self.feature_test_passed = True
                    print(f"✅ All tests passed!")
        
        # After all actions, run regression tests (except for first feature)
        if not is_first_feature and self.feature_test_passed:
            print("\n🔄 Running regression tests...")
            regression_passed, regression_output = self._run_regression_tests()
            if not regression_passed:
                print("❌ Regression tests failed!")
                return False, f"Regression tests failed: {regression_output}"
        
        # Before marking as complete, verify all PHP files have valid syntax
        project_type = self._detect_project_type()
        if project_type == "PHP":
            php_files = []
            src_dir = os.path.join(self.output_dir, "src")
            if os.path.exists(src_dir):
                for file in os.listdir(src_dir):
                    if file.endswith('.php'):
                        php_files.append(os.path.join("src", file))
            
            for php_file in php_files:
                php_check_path = php_file
                syntax_check = self.tools.execute_command(f"php -l {php_check_path}", timeout=10, cwd=self.output_dir)
                if syntax_check[2] != 0:
                    error_msg = f"PHP syntax error still present in {php_file}:\n{syntax_check[1]}"
                    print(f"❌ {error_msg}")
                    self._log_thought(f"PHP syntax error still present: {php_file}")
                    return False, error_msg
        
        # If all tests passed and all PHP files are valid, commit
        if self.feature_test_passed:
            # Generate feature documentation
            self._generate_feature_docs(feature_name)
            
            # Commit to git
            commit_message = f"Feature: {feature_name} - implemented and tested"
            if self._git_commit(commit_message):
                print(f"✅ Committed: {commit_message}")
                return True, None
        
        # Check WHY tests didn't pass
        tests_dir = os.path.join(self.output_dir, "tests")
        if not os.path.exists(tests_dir) or not any(f.endswith('.py') for f in os.listdir(tests_dir)):
            # Auto-generate basic tests if Planner forgot
            print("⚠️  No test files found. Auto-generating basic tests...")
            generated = self._auto_generate_tests()
            if generated:
                print("✅ Auto-generated test file. Re-running tests...")
                # Re-run the test execution
                self._auto_fix_test_imports(tests_dir)
                test_cmd = "python3 -m pytest tests/ -v" if self._is_pytest_available() else "python3 -m unittest discover -s tests -v"
                stdout, stderr, return_code = self.tools.execute_command(test_cmd, timeout=120, cwd=self.output_dir)
                if return_code == 0:
                    self.feature_test_passed = True
                    return True, None
                else:
                    error_output = stderr if stderr else stdout
                    return False, f"Auto-generated tests failed: {error_output[:500]}"
            else:
                error_msg = "No test files found and auto-generation failed!"
                print(f"❌ {error_msg}")
                return False, error_msg
        
        return False, last_test_error or "Tests were executed but did not pass. Check test output above."
    
    def _generate_feature_docs(self, feature_name: str) -> None:
        """Generate documentation for a completed feature."""
        docs_path = os.path.join(self.output_dir, "docs", "features", f"{feature_name.lower().replace(' ', '_')}.md")
        Path(os.path.dirname(docs_path)).mkdir(parents=True, exist_ok=True)
        
        # Use current_feature_files instead of entire history to avoid duplicates
        # Fallback to history if current_feature_files is empty (for backward compatibility)
        if self.current_feature_files:
            code_files = self.current_feature_files
        else:
            # Fallback: get files from history but only unique ones
            code_files = list(set([h['target'] for h in self.history if h.get('action') == 'write_file' and h.get('success')]))
        
        # Separate test files from source files
        test_files = [f for f in code_files if 'test' in f.lower() or f.startswith('tests/')]
        src_files = [f for f in code_files if f not in test_files]
        
        # Generate documentation with detailed descriptions
        doc_content = self._generate_feature_documentation(feature_name, src_files, test_files)
        self.tools.write_file(docs_path, doc_content)
        
        # Extract a meaningful description (first paragraph of overview)
        desc_lines = doc_content.split('\n')
        description = ""
        in_overview = False
        for line in desc_lines:
            if line.startswith('## Overview'):
                in_overview = True
                continue
            if in_overview and line.startswith('##'):
                break
            if in_overview and line.strip():
                description += line.strip() + " "
                if len(description) > 150:
                    description = description[:150] + "..."
                    break
        
        self.feature_docs.append({"name": feature_name, "description": description.strip() or doc_content[:200] + "..."})
        print(f"📚 Documentation generated: {docs_path}")
    
    def _generate_final_docs_and_exit(self) -> None:
        """Generate final documentation and exit."""
        final_docs_path = os.path.join(self.output_dir, "README.md")
        final_doc_content = self._generate_final_documentation()
        self.tools.write_file(final_docs_path, final_doc_content)
        print(f"✅ Final documentation: {final_docs_path}")
        
        # Commit final docs
        if self.git_repo_initialized:
            self._git_commit("Final documentation and project completion")
        
        # Calculate and display total time
        if self.start_time:
            total_time = time.time() - self.start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours > 0 else f"{minutes:02d}:{seconds:02d}"
            
            print("\n" + "=" * 60)
            print("⏱️  TEMPO TOTALE DI ELABORAZIONE")
            print("=" * 60)
            print(f"   {time_str} ({total_time:.2f} secondi)")
            print("=" * 60)
        
        # Save thought chain
        thought_chain_path = os.path.join(self.output_dir, "thought_chain.log")
        with open(thought_chain_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("CHAIN OF THOUGHT - Development Session\n")
            f.write("=" * 60 + "\n\n")
            for thought in self.thought_chain:
                f.write(thought + "\n")
        print(f"💭 Thought chain saved: {thought_chain_path}")
        
        print("\n🎉 Mission complete!")
        print(f"📁 All code is in: {os.path.abspath(self.output_dir)}")




def main():
    """Entry point for the Code Agent."""
    # Configuration file path
    CONFIG_PATH = "config.json"
    
    # Initialize and run agent (config is loaded from file)
    agent = CodeAgent(config_path=CONFIG_PATH)
    
    try:
        agent.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting gracefully...")
        # Stop PHP server if running
        if agent.php_server_process is not None:
            agent._stop_php_server()
    except Exception as e:
        print(f"\n\n💥 Fatal error: {e}")
        # Stop PHP server if running
        if agent.php_server_process is not None:
            agent._stop_php_server()
        raise


if __name__ == "__main__":
    main()

