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
    
    def __init__(self, api_url: str, model_name: str = "local-model", timeout: int = 240, temperature: float = 0.2):
        """
        Initialize LLM client.
        
        Args:
            api_url: Base URL for the API endpoint
            model_name: Name of the model to use
            timeout: Request timeout in seconds (default 240s for large models)
            temperature: Sampling temperature (low for coding, medium for planning)
        """
        self.api_url = api_url
        self.model_name = model_name
        self.timeout = timeout
        self.temperature = temperature
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        """
        Send chat completion request to the LLM API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature if provided
            
        Returns:
            Response content as string
            
        Raises:
            requests.exceptions.RequestException: On connection/timeout errors
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": False
        }
        
        try:
            print(f"🌐 Sending request to {self.api_url} (timeout: {self.timeout}s)...")
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
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
            temperature=planner_config.get('temperature', 0.7)
        )
        self.executor_client = LLMClient(
            api_url=executor_config.get('server', 'http://192.168.1.29:8080'),
            model_name=executor_config.get('model', 'local-model'),
            timeout=executor_config.get('timeout', 240),
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
                "timeout": 120,
                "temperature": 0.7
            },
            "executor": {
                "server": "http://192.168.1.29:8080",
                "model": "local-model",
                "timeout": 240,
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
        
        # If text doesn't end with ']', it's likely truncated
        if not text.endswith(']'):
            # Count open brackets and braces
            open_brackets = text.count('[') - text.count(']')
            open_braces = text.count('{') - text.count('}')
            
            # Find the last position where we might have been in a string
            # Look for incomplete strings (odd number of quotes)
            quote_positions = [m.start() for m in re.finditer(r'(?<!\\)"', text)]
            
            # If we have an odd number of quotes, we're inside a string
            if len(quote_positions) % 2 == 1:
                # Find the last quote and close the string
                last_quote_idx = quote_positions[-1]
                # Check if there's content after the last quote that needs to be closed
                after_quote = text[last_quote_idx + 1:]
                # If there's content, close the string
                if after_quote.strip() and not after_quote.strip().startswith(','):
                    # Insert closing quote
                    text = text[:last_quote_idx + 1] + '"' + text[last_quote_idx + 1:]
            
            # Find last complete object/array
            # Try to find the last complete JSON object
            brace_stack = []
            last_valid_pos = -1
            
            for i, char in enumerate(text):
                if char == '{':
                    brace_stack.append(i)
                elif char == '}':
                    if brace_stack:
                        brace_stack.pop()
                        if not brace_stack:  # All braces closed
                            last_valid_pos = i
            
            # If we found a complete object, close everything after it
            if last_valid_pos != -1 and last_valid_pos < len(text) - 1:
                # Keep everything up to last valid position, close the rest
                text = text[:last_valid_pos + 1]
                # Close remaining braces
                text += '}' * open_braces
                # Close remaining brackets
                text += ']' * open_brackets
            else:
                # No complete object found, try to close everything
                # Close remaining braces first
                text += '}' * open_braces
                # Close remaining brackets
                text += ']' * open_brackets
        
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
            # Include task context to help Executor understand requirements
            task_context = ""
            if hasattr(self, 'task_description') and self.task_description:
                # Extract relevant parts of task for context (first 1500 chars to avoid token limits)
                task_context = f"\n\nTASK CONTEXT (for reference):\n{self.task_description[:1500]}"
                if len(self.task_description) > 1500:
                    task_context += "\n[... task description continues ...]"
            
            prompt = f"""You are an expert Developer. Your Task: Write the content for the file '{target_file}'.

INSTRUCTIONS FROM PLANNER:
{instruction}
{task_context}

REQUIREMENTS: 
- Return ONLY the raw code. Do NOT use markdown. Do NOT write conversational text.
- Follow the task requirements exactly as specified in the TASK CONTEXT above.
- Use the correct database technology (SQLite for PHP projects, not MySQLi).
- Implement all required functionality as described in the task."""
        
        messages = [
            {"role": "system", "content": "You are an expert developer. Always return only raw code, no markdown, no explanations."},
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
        Generate a coherence report analyzing frontend-backend consistency.
        
        Args:
            project_type: Type of project (PHP, Python, etc.)
            
        Returns:
            Coherence report string with detected issues
        """
        if project_type != "PHP":
            return ""  # Only for PHP projects with frontend-backend separation
        
        import re
        issues = []
        warnings = []
        
        # Find HTML files
        src_dir = os.path.join(self.output_dir, "src")
        if not os.path.exists(src_dir):
            return ""
        
        html_files = [f for f in os.listdir(src_dir) if f.endswith('.html')]
        php_files = [f for f in os.listdir(src_dir) if f.endswith('.php')]
        
        if not html_files or not php_files:
            return ""
        
        # Extract frontend API calls
        frontend_calls = []
        for html_file in html_files:
            html_path = os.path.join(src_dir, html_file)
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_lines = f.readlines()
                calls = self._extract_frontend_api_calls(html_lines)
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
                        api_lines = f.readlines()
                    endpoints = self._extract_api_endpoints(api_lines)
                    backend_endpoints.extend(endpoints)
                except:
                    pass
                break
        
        # Compare frontend calls with backend endpoints
        if frontend_calls and backend_endpoints:
            report_parts = ["\n🔍 COHERENCE ANALYSIS (Frontend-Backend):"]
            
            # Check for endpoint mismatches
            frontend_actions = {call['action']: call for call in frontend_calls if call.get('action')}
            backend_actions = {ep['action']: ep for ep in backend_endpoints}
            
            # Missing endpoints
            missing = set(frontend_actions.keys()) - set(backend_actions.keys())
            if missing:
                for action in missing:
                    call = frontend_actions[action]
                    issues.append(f"❌ Frontend calls '{action}' but backend has no handler")
                    report_parts.append(f"  ❌ MISSING ENDPOINT: Frontend calls action='{action}' ({call['method']}) but backend doesn't handle it")
            
            # Extra endpoints (not called by frontend)
            extra = set(backend_actions.keys()) - set(frontend_actions.keys())
            if extra:
                for action in extra:
                    warnings.append(f"⚠️ Backend has '{action}' endpoint but frontend doesn't call it")
                    report_parts.append(f"  ⚠️ UNUSED ENDPOINT: Backend handles '{action}' but frontend doesn't call it")
            
            # Method mismatches
            for action in set(frontend_actions.keys()) & set(backend_actions.keys()):
                frontend_call = frontend_actions[action]
                backend_ep = backend_actions[action]
                if frontend_call['method'] != backend_ep['method']:
                    issues.append(f"❌ Method mismatch for '{action}': frontend uses {frontend_call['method']}, backend expects {backend_ep['method']}")
                    report_parts.append(f"  ❌ METHOD MISMATCH: '{action}' - Frontend: {frontend_call['method']}, Backend: {backend_ep['method']}")
            
            # JSON format mismatches
            for action in set(frontend_actions.keys()) & set(backend_actions.keys()):
                frontend_call = frontend_actions[action]
                backend_ep = backend_actions[action]
                frontend_keys = set(frontend_call.get('expected_response_keys', []))
                backend_keys = set(backend_ep.get('response_keys', []))
                if frontend_keys and backend_keys:
                    missing_keys = frontend_keys - backend_keys
                    extra_keys = backend_keys - frontend_keys
                    if missing_keys or extra_keys:
                        warnings.append(f"⚠️ JSON format mismatch for '{action}'")
                        if missing_keys:
                            report_parts.append(f"  ⚠️ JSON MISMATCH '{action}': Frontend expects keys {missing_keys} but backend doesn't return them")
                        if extra_keys:
                            report_parts.append(f"  ⚠️ JSON MISMATCH '{action}': Backend returns keys {extra_keys} but frontend doesn't use them")
            
            # Check for require/include mismatches
            if api_file:
                api_path = os.path.join(src_dir, api_file)
                try:
                    with open(api_path, 'r', encoding='utf-8') as f:
                        api_content = f.read()
                    require_matches = re.findall(r'require\s+[\'"]([^\'"]+\.php)[\'"]', api_content)
                    for req_file in require_matches:
                        req_path = os.path.join(src_dir, req_file)
                        if not os.path.exists(req_path):
                            # Check if similar file exists
                            similar_files = [f for f in php_files if req_file.replace('.php', '') in f or f.replace('.php', '') in req_file]
                            if similar_files:
                                issues.append(f"❌ api.php requires '{req_file}' but file doesn't exist. Similar file found: {similar_files[0]}")
                                report_parts.append(f"  ❌ DEPENDENCY ERROR: api.php requires '{req_file}' but it doesn't exist. Did you mean '{similar_files[0]}'?")
                            else:
                                issues.append(f"❌ api.php requires '{req_file}' but file doesn't exist")
                                report_parts.append(f"  ❌ DEPENDENCY ERROR: api.php requires '{req_file}' but file doesn't exist")
                except:
                    pass
            
            if issues or warnings:
                report_parts.insert(1, f"\n  Found {len(issues)} critical issues and {len(warnings)} warnings:")
                return '\n'.join(report_parts)
        
        return ""
    
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
            # Prefer pytest if test files exist
            if py_test_files:
                test_commands.extend([
                    "pytest",
                    "python -m pytest",
                    "python -m pytest tests/",
                    "pytest tests/",
                ])
            # Fallback to unittest
            test_commands.extend([
                "python -m unittest discover",
                "python -m unittest discover -s tests",
                "python -m unittest",
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
        
        Args:
            plan: Execution plan from Planner
            project_type: Type of project (PHP, Python, etc.)
            
        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []
        
        if project_type != "PHP":
            return True, warnings
        
        # Extract files that will be written
        files_to_write = []
        for action in plan:
            if action.get('action') == 'write_file':
                target = action.get('target', '')
                if target:
                    files_to_write.append(target)
        
        # Check for require/include mismatches in PHP files
        src_dir = os.path.join(self.output_dir, "src")
        if os.path.exists(src_dir):
            existing_php_files = [f for f in os.listdir(src_dir) if f.endswith('.php')]
            
            for action in plan:
                if action.get('action') == 'write_file':
                    target = action.get('target', '')
                    if target.endswith('.php'):
                        # Check if instruction mentions require/include
                        instruction = action.get('content_instruction', '')
                        import re
                        # Capture ALL require/include statements, not just .php files
                        require_matches = re.findall(r'(?:require|include)(?:_once)?\s+[\'"]([^\'"]+)[\'"]', instruction)
                        for req_file in require_matches:
                            # Check if it's a valid PHP file (should end with .php)
                            if not req_file.endswith('.php'):
                                # Invalid: trying to require/include non-PHP file
                                if req_file.endswith(('.sqlite', '.db', '.sqlite3', '.json', '.txt', '.log')):
                                    warnings.append(f"❌ CRITICAL: Plan will write {target} requiring '{req_file}' which is a data file, not a PHP file! Cannot require/include database or data files.")
                                else:
                                    warnings.append(f"⚠️ Plan will write {target} requiring '{req_file}' which doesn't end with .php. This may be an error.")
                            
                            # Check if file exists (only for .php files)
                            if req_file.endswith('.php'):
                                req_path = os.path.join(src_dir, req_file)
                                if not os.path.exists(req_path):
                                    # Check for similar files
                                    similar = [f for f in existing_php_files if req_file.replace('.php', '') in f or f.replace('.php', '') in req_file]
                                    if similar:
                                        warnings.append(f"⚠️ Plan will write {target} requiring '{req_file}' but file doesn't exist. Did you mean '{similar[0]}'?")
                                    else:
                                        warnings.append(f"⚠️ Plan will write {target} requiring '{req_file}' but file doesn't exist")
        
        return len(warnings) == 0, warnings
    
    def _validate_generated_code(self, project_type: str) -> tuple[bool, List[str]]:
        """
        Validate generated code for coherence issues after execution.
        
        Args:
            project_type: Type of project (PHP, Python, etc.)
            
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        if project_type != "PHP":
            return True, errors
        
        # Check for require/include mismatches
        src_dir = os.path.join(self.output_dir, "src")
        if not os.path.exists(src_dir):
            return True, errors
        
        php_files = [f for f in os.listdir(src_dir) if f.endswith('.php')]
        
        for php_file in php_files:
            php_path = os.path.join(src_dir, php_file)
            try:
                with open(php_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                import re
                # Capture ALL require/include statements, not just .php files
                require_matches = re.findall(r'(?:require|include)(?:_once)?\s+[\'"]([^\'"]+)[\'"]', content)
                for req_file in require_matches:
                    # Check if it's a valid PHP file (should end with .php)
                    if not req_file.endswith('.php'):
                        # Invalid: trying to require/include non-PHP file
                        if req_file.endswith(('.sqlite', '.db', '.sqlite3', '.json', '.txt', '.log', '.csv')):
                            errors.append(f"❌ CRITICAL: {php_file} requires '{req_file}' which is a data file, not a PHP file! Cannot require/include database or data files. This will cause a fatal error.")
                        elif req_file.endswith(('.html', '.css', '.js')):
                            errors.append(f"❌ {php_file} requires '{req_file}' which is not a PHP file. Use readfile() or include HTML differently.")
                        else:
                            errors.append(f"⚠️ {php_file} requires '{req_file}' which doesn't end with .php. This may be an error.")
                    
                    # Check if file exists (only for .php files)
                    if req_file.endswith('.php'):
                        req_path = os.path.join(src_dir, req_file)
                        if not os.path.exists(req_path):
                            # Check for similar files
                            similar = [f for f in php_files if req_file.replace('.php', '') in f or f.replace('.php', '') in req_file]
                            if similar:
                                errors.append(f"❌ {php_file} requires '{req_file}' but file doesn't exist. Did you mean '{similar[0]}'?")
                            else:
                                errors.append(f"❌ {php_file} requires '{req_file}' but file doesn't exist")
            except Exception as e:
                pass
        
        # Check coherence report
        coherence_report = self._generate_coherence_report(project_type)
        if coherence_report and '❌' in coherence_report:
            # Extract critical issues
            import re
            critical_issues = re.findall(r'❌ ([^\n]+)', coherence_report)
            errors.extend([f"❌ {issue}" for issue in critical_issues])
        
        return len(errors) == 0, errors
    
    def run(self) -> None:
        """
        Main execution loop with clear feature-by-feature workflow.
        
        Flow:
        1. Planner creates list of features
        2. For each feature:
           a. Planner creates test list
           b. Executor writes code
           c. Run all tests
           d. If all pass: create git repo (if needed) and commit
           e. Move to next feature
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
        
        # Step 2: Get feature list from Planner
        print("\n" + "=" * 60)
        print("📋 STEP 1: Getting feature list from Planner...")
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
            
            # Process feature until all tests pass
            max_attempts = 10
            attempt = 0
            feature_complete = False
            
            last_test_error = None
            while not feature_complete and attempt < max_attempts:
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
                    last_test_error = test_error
                    print(f"\n❌ Feature '{feature_name}' attempt {attempt} failed. Retrying...")
                    time.sleep(2)
            
            if not feature_complete:
                print(f"\n⚠️  Feature '{feature_name}' failed after {max_attempts} attempts. Continuing to next feature...")
        
        # Final step: Generate final documentation
        print("\n" + "=" * 60)
        print("📚 Generating final documentation...")
        print("=" * 60)
        self._generate_final_docs_and_exit()
        
        # Cleanup: Stop PHP server if running
        self._stop_php_server()
    
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
        
        # CRITICAL: Add information about existing files so Planner knows what's already implemented
        existing_files_context = self._get_existing_files_context(project_type)
        context += existing_files_context
        
        # Add explicit coherence rules
        context += "\n\n🚨 CRITICAL COHERENCE RULES - YOU MUST FOLLOW THESE:\n\n"
        context += "1. FILE DEPENDENCIES:\n"
        context += "   - Check ALL require/include statements in existing files above\n"
        context += "   - If a file requires 'database.php' but 'db.php' exists, you MUST use 'db.php' instead\n"
        context += "   - If dependency status shows ❌ MISSING, you MUST fix the require statement to use existing file\n"
        context += "   - DO NOT create new files with different names if similar files exist\n\n"
        
        context += "2. API ENDPOINT MATCHING:\n"
        context += "   - If COHERENCE ANALYSIS shows mismatches, you MUST fix them\n"
        context += "   - Frontend action names MUST exactly match backend case statements\n"
        context += "   - HTTP methods MUST match (GET vs POST)\n"
        context += "   - If frontend calls 'getSeats' but backend has 'get_seats', you MUST make them match\n\n"
        
        context += "3. JSON FORMAT CONSISTENCY:\n"
        context += "   - Frontend expected response keys MUST match backend return keys\n"
        context += "   - If frontend expects {success: true, seats: []}, backend MUST return exactly that structure\n"
        context += "   - Check COHERENCE ANALYSIS for JSON format mismatches and fix them\n\n"
        
        context += "4. BEFORE WRITING api.php:\n"
        context += "   - Read ALL HTML files to see all fetch() calls\n"
        context += "   - Extract exact endpoint names, HTTP methods, and expected JSON formats\n"
        context += "   - Ensure api.php handles ALL endpoints the frontend calls\n"
        context += "   - Ensure response JSON structure matches what frontend expects\n\n"
        
        context += "5. BEFORE WRITING ANY FILE:\n"
        context += "   - Check if similar file exists (e.g., db.php vs database.php)\n"
        context += "   - Use existing file names and structures\n"
        context += "   - Maintain consistency with existing code patterns\n\n"
        
        if last_test_error:
            context += f"⚠️ LAST ERROR:\n{last_test_error}\n\n"
            context += "INSTRUCTIONS:\n"
            # Check if error is about PHP syntax errors (not test errors)
            if "PHP syntax error" in last_test_error or "Parse error" in last_test_error or "syntax error" in last_test_error.lower():
                context += "🚨 CRITICAL: This is a PHP SYNTAX ERROR, NOT a test error!\n"
                context += "1. You MUST fix the PHP source file that has the syntax error\n"
                context += "2. The error message above tells you which file and which line has the problem\n"
                context += "3. FIRST: Use read_file action to read the existing file with the error\n"
                context += "4. THEN: Use write_file action to write the CORRECTED version of the SAME file\n"
                context += "5. DO NOT create new files - you MUST fix the existing file\n"
                context += "6. After fixing the PHP file, verify it has valid syntax\n"
                context += "7. Only then create/run tests if needed\n"
                context += "8. DO NOT create test files until the PHP syntax error is fixed\n"
                # Extract file path from error message
                import re
                file_match = re.search(r'in (src/[^\s]+\.php)', last_test_error)
                if file_match:
                    error_file = file_match.group(1)
                    context += f"\n⚠️ The file with the error is: {error_file}\n"
                    context += f"You MUST read this file first (use read_file action), then fix it (use write_file action). Do NOT create other files.\n"
            else:
                context += "1. Fix ONLY the test files that failed (do NOT regenerate source code files that already exist)\n"
            if project_type == "PHP":
                context += "2. For PHP projects: Generate PYTHON test files (test_*.py) that use curl/requests to test API endpoints\n"
            context += "3. Execute the fixed tests\n"
        elif is_first_feature:
            context += "INSTRUCTIONS - FOLLOW THIS EXACT ORDER:\n"
            context += "1. Create source code files FIRST (setup, database, API, etc.) - place ALL files in 'src/' directory (no subdirectories)\n"
            context += "2. Create test files AFTER source files exist\n"
            if project_type == "PHP":
                context += "   - For PHP: Create PYTHON test files (test_*.py) in 'tests/' directory\n"
                context += "   - Use Python's requests library or subprocess with curl to test PHP API endpoints\n"
                context += "   - Test by making HTTP requests to the PHP application\n"
            context += "3. Execute all tests (verify files exist before testing)\n"
            context += "   - CRITICAL: After writing each test file, you MUST include an execute_command action to run it\n"
            context += "   - Example: After writing tests/test_setup.py, add: {\"action\": \"execute_command\", \"target\": \"python3 tests/test_setup.py\"}\n"
            context += "   - For PHP projects: Use 'python3 tests/test_*.py' to execute Python tests\n"
            context += "4. If all tests pass, create git repository and commit\n"
        else:
            context += "INSTRUCTIONS - FOLLOW THIS EXACT ORDER:\n"
            context += "1. Create source code files for this feature FIRST - place ALL files in 'src/' directory (no subdirectories)\n"
            context += "2. Create test files AFTER source files exist\n"
            if project_type == "PHP":
                context += "   - For PHP: Create PYTHON test files (test_*.py) in 'tests/' directory\n"
                context += "   - Use Python's requests library or subprocess with curl to test PHP API endpoints\n"
                context += "   - Test by making HTTP requests to the PHP application\n"
            context += "3. Execute feature tests (verify files exist before testing)\n"
            context += "   - CRITICAL: After writing each test file, you MUST include an execute_command action to run it\n"
            context += "   - Example: After writing tests/test_api.py, add: {\"action\": \"execute_command\", \"target\": \"python3 tests/test_api.py\"}\n"
            context += "   - For PHP projects: Use 'python3 tests/test_*.py' to execute Python tests\n"
            context += "4. Execute ALL regression tests (full test suite)\n"
            context += "5. If all tests pass, commit to git\n"
        
        context += "\nOUTPUT FORMAT: JSON array of actions. Schema:\n"
        context += '[{"step": int, "action": "read_file"|"write_file"|"execute_command", "target": "path", "content_instruction": "instruction"}]'
        context += "\n\nAVAILABLE ACTIONS:\n"
        context += "- read_file: Read an existing file to see its content (use this BEFORE fixing a file with errors)\n"
        context += "- write_file: Write or overwrite a file\n"
        context += "- execute_command: Execute a shell command\n"
        context += "\n⚠️ CRITICAL: content_instruction must be DETAILED and SPECIFIC:\n"
        context += "- Include key requirements from the TASK (database type, API endpoints, data structures, etc.)\n"
        context += "- Don't just say 'write API handler' - specify which endpoints, what they do, what database to use\n"
        context += "- Example BAD: 'Write API handler for AJAX requests'\n"
        context += "- Example GOOD: 'Write API handler using PDO/SQLite (NOT MySQLi) that handles: login (session-based), register (with email verification token), get_seats (returns seat availability for a date), book_seat (creates booking with seat_number and booking_date), admin_actions (set total_seats in config table, get rooming list for date). Database file is database.sqlite. Use session_start() for authentication.'\n"
        context += "- For setup.php: Specify exact table schemas with all required columns from the TASK\n"
        context += "- For db.php: Specify SQLite connection (not MySQLi), PDO usage\n"
        
        # CRITICAL: For PHP projects, explicitly forbid PHP test files
        if project_type == "PHP":
            context += "\n\n🚫 FORBIDDEN FOR PHP PROJECTS:\n"
            context += "- Do NOT create test files with .php extension (e.g., tests/test_*.php)\n"
            context += "- Do NOT create test files named test_*.php\n"
            context += "- ONLY create Python test files (test_*.py)\n"
            context += "- If you see 'test' in target path and it ends with .php, change it to .py\n"
        
        messages = [
            {"role": "system", "content": "You are a Senior Software Architect. Create detailed execution plans. Respond with JSON arrays only. For PHP projects, you MUST generate Python test files (.py), NEVER PHP test files (.php). When writing content_instruction, be SPECIFIC and include all key requirements from the TASK."},
            {"role": "user", "content": context}
        ]
        
        try:
            response = self.planner_client.chat_completion(messages, temperature=0.7)
            plan = self._clean_json(response)
            return plan
        except Exception as e:
            print(f"⚠️  Error getting feature plan: {e}")
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
                    code_content = self._ask_executor(content_instruction, normalized_path)
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
        # If so, execute them automatically
        project_type = self._detect_project_type()
        if project_type == "PHP":
            # Find all Python test files in tests/ directory
            tests_dir = os.path.join(self.output_dir, "tests")
            if os.path.exists(tests_dir):
                test_files = [f for f in os.listdir(tests_dir) if f.endswith('.py') and (f.startswith('test_') or 'test' in f.lower())]
                if test_files and not self.feature_test_passed:
                    # Tests were written but not executed - execute them now
                    print(f"\n⚠️  Test files were written but not executed. Executing them now...")
                    for test_file in test_files:
                        test_path = os.path.join("tests", test_file)
                        if not self._start_php_server():
                            return False, "Failed to start PHP server for testing"
                        
                        # Validate Python syntax first
                        syntax_check = self.tools.execute_command(f"python3 -m py_compile {test_path}", timeout=10, cwd=self.output_dir)
                        if syntax_check[2] != 0:
                            error_msg = f"Python syntax error in {test_path}:\n{syntax_check[1]}"
                            print(f"❌ {error_msg}")
                            return False, error_msg
                        
                        print(f"🧪 Executing test: {test_path}")
                        stdout, stderr, return_code = self.tools.execute_command(f"python3 {test_path}", timeout=60, cwd=self.output_dir)
                        self.test_counter += 1
                        status = "PASSED" if return_code == 0 else "FAILED"
                        print(f"🧪 Test #{self.test_counter}: {test_path} - {status}")
                        self._log_thought(f"Test #{self.test_counter}: {test_path} - {status}")
                        
                        if return_code != 0:
                            error_msg = f"Test failed: {test_path}\nSTDOUT: {stdout}\nSTDERR: {stderr}"
                            print(f"❌ Test failed. Output:\n{stderr}")
                            return False, error_msg
                        else:
                            self.feature_test_passed = True
        
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
        
        return False, last_test_error or "Tests did not pass"
    
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
        
        # CRITICAL: Add information about existing files so Planner knows what's already implemented
        existing_files_context = self._get_existing_files_context(project_type)
        context += existing_files_context
        
        # Add explicit coherence rules
        context += "\n\n🚨 CRITICAL COHERENCE RULES - YOU MUST FOLLOW THESE:\n\n"
        context += "1. FILE DEPENDENCIES:\n"
        context += "   - Check ALL require/include statements in existing files above\n"
        context += "   - If a file requires 'database.php' but 'db.php' exists, you MUST use 'db.php' instead\n"
        context += "   - If dependency status shows ❌ MISSING, you MUST fix the require statement to use existing file\n"
        context += "   - DO NOT create new files with different names if similar files exist\n\n"
        
        context += "2. API ENDPOINT MATCHING:\n"
        context += "   - If COHERENCE ANALYSIS shows mismatches, you MUST fix them\n"
        context += "   - Frontend action names MUST exactly match backend case statements\n"
        context += "   - HTTP methods MUST match (GET vs POST)\n"
        context += "   - If frontend calls 'getSeats' but backend has 'get_seats', you MUST make them match\n\n"
        
        context += "3. JSON FORMAT CONSISTENCY:\n"
        context += "   - Frontend expected response keys MUST match backend return keys\n"
        context += "   - If frontend expects {success: true, seats: []}, backend MUST return exactly that structure\n"
        context += "   - Check COHERENCE ANALYSIS for JSON format mismatches and fix them\n\n"
        
        context += "4. BEFORE WRITING api.php:\n"
        context += "   - Read ALL HTML files to see all fetch() calls\n"
        context += "   - Extract exact endpoint names, HTTP methods, and expected JSON formats\n"
        context += "   - Ensure api.php handles ALL endpoints the frontend calls\n"
        context += "   - Ensure response JSON structure matches what frontend expects\n\n"
        
        context += "5. BEFORE WRITING ANY FILE:\n"
        context += "   - Check if similar file exists (e.g., db.php vs database.php)\n"
        context += "   - Use existing file names and structures\n"
        context += "   - Maintain consistency with existing code patterns\n\n"
        
        if last_test_error:
            context += f"⚠️ LAST ERROR:\n{last_test_error}\n\n"
            context += "INSTRUCTIONS:\n"
            # Check if error is about PHP syntax errors (not test errors)
            if "PHP syntax error" in last_test_error or "Parse error" in last_test_error or "syntax error" in last_test_error.lower():
                context += "🚨 CRITICAL: This is a PHP SYNTAX ERROR, NOT a test error!\n"
                context += "1. You MUST fix the PHP source file that has the syntax error\n"
                context += "2. The error message above tells you which file and which line has the problem\n"
                context += "3. FIRST: Use read_file action to read the existing file with the error\n"
                context += "4. THEN: Use write_file action to write the CORRECTED version of the SAME file\n"
                context += "5. DO NOT create new files - you MUST fix the existing file\n"
                context += "6. After fixing the PHP file, verify it has valid syntax\n"
                context += "7. Only then create/run tests if needed\n"
                context += "8. DO NOT create test files until the PHP syntax error is fixed\n"
                # Extract file path from error message
                import re
                file_match = re.search(r'in (src/[^\s]+\.php)', last_test_error)
                if file_match:
                    error_file = file_match.group(1)
                    context += f"\n⚠️ The file with the error is: {error_file}\n"
                    context += f"You MUST read this file first (use read_file action), then fix it (use write_file action). Do NOT create other files.\n"
            else:
                context += "1. Fix ONLY the test files that failed (do NOT regenerate source code files that already exist)\n"
            if project_type == "PHP":
                context += "2. For PHP projects: Generate PYTHON test files (test_*.py) that use curl/requests to test API endpoints\n"
            context += "3. Execute the fixed tests\n"
        elif is_first_feature:
            context += "INSTRUCTIONS - FOLLOW THIS EXACT ORDER:\n"
            context += "1. Create source code files FIRST (setup, database, API, etc.) - place ALL files in 'src/' directory (no subdirectories)\n"
            context += "2. Create test files AFTER source files exist\n"
            if project_type == "PHP":
                context += "   - For PHP: Create PYTHON test files (test_*.py) in 'tests/' directory\n"
                context += "   - Use Python's requests library or subprocess with curl to test PHP API endpoints\n"
                context += "   - Test by making HTTP requests to the PHP application\n"
            context += "3. Execute all tests (verify files exist before testing)\n"
            context += "   - CRITICAL: After writing each test file, you MUST include an execute_command action to run it\n"
            context += "   - Example: After writing tests/test_setup.py, add: {\"action\": \"execute_command\", \"target\": \"python3 tests/test_setup.py\"}\n"
            context += "   - For PHP projects: Use 'python3 tests/test_*.py' to execute Python tests\n"
            context += "4. If all tests pass, create git repository and commit\n"
        else:
            context += "INSTRUCTIONS - FOLLOW THIS EXACT ORDER:\n"
            context += "1. Create source code files for this feature FIRST - place ALL files in 'src/' directory (no subdirectories)\n"
            context += "2. Create test files AFTER source files exist\n"
            if project_type == "PHP":
                context += "   - For PHP: Create PYTHON test files (test_*.py) in 'tests/' directory\n"
                context += "   - Use Python's requests library or subprocess with curl to test PHP API endpoints\n"
                context += "   - Test by making HTTP requests to the PHP application\n"
            context += "3. Execute feature tests (verify files exist before testing)\n"
            context += "   - CRITICAL: After writing each test file, you MUST include an execute_command action to run it\n"
            context += "   - Example: After writing tests/test_api.py, add: {\"action\": \"execute_command\", \"target\": \"python3 tests/test_api.py\"}\n"
            context += "   - For PHP projects: Use 'python3 tests/test_*.py' to execute Python tests\n"
            context += "4. Execute ALL regression tests (full test suite)\n"
            context += "5. If all tests pass, commit to git\n"
        
        context += "\nOUTPUT FORMAT: JSON array of actions. Schema:\n"
        context += '[{"step": int, "action": "read_file"|"write_file"|"execute_command", "target": "path", "content_instruction": "instruction"}]'
        context += "\n\nAVAILABLE ACTIONS:\n"
        context += "- read_file: Read an existing file to see its content (use this BEFORE fixing a file with errors)\n"
        context += "- write_file: Write or overwrite a file\n"
        context += "- execute_command: Execute a shell command\n"
        context += "\n⚠️ CRITICAL: content_instruction must be DETAILED and SPECIFIC:\n"
        context += "- Include key requirements from the TASK (database type, API endpoints, data structures, etc.)\n"
        context += "- Don't just say 'write API handler' - specify which endpoints, what they do, what database to use\n"
        context += "- Example BAD: 'Write API handler for AJAX requests'\n"
        context += "- Example GOOD: 'Write API handler using PDO/SQLite (NOT MySQLi) that handles: login (session-based), register (with email verification token), get_seats (returns seat availability for a date), book_seat (creates booking with seat_number and booking_date), admin_actions (set total_seats in config table, get rooming list for date). Database file is database.sqlite. Use session_start() for authentication.'\n"
        context += "- For setup.php: Specify exact table schemas with all required columns from the TASK\n"
        context += "- For db.php: Specify SQLite connection (not MySQLi), PDO usage\n"
        
        # CRITICAL: For PHP projects, explicitly forbid PHP test files
        if project_type == "PHP":
            context += "\n\n🚫 FORBIDDEN FOR PHP PROJECTS:\n"
            context += "- Do NOT create test files with .php extension (e.g., tests/test_*.php)\n"
            context += "- Do NOT create test files named test_*.php\n"
            context += "- ONLY create Python test files (test_*.py)\n"
            context += "- If you see 'test' in target path and it ends with .php, change it to .py\n"
        
        messages = [
            {"role": "system", "content": "You are a Senior Software Architect. Create detailed execution plans. Respond with JSON arrays only. For PHP projects, you MUST generate Python test files (.py), NEVER PHP test files (.php). When writing content_instruction, be SPECIFIC and include all key requirements from the TASK."},
            {"role": "user", "content": context}
        ]
        
        try:
            response = self.planner_client.chat_completion(messages, temperature=0.7)
            plan = self._clean_json(response)
            return plan
        except Exception as e:
            print(f"⚠️  Error getting feature plan: {e}")
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
                    code_content = self._ask_executor(content_instruction, normalized_path)
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
        # If so, execute them automatically
        project_type = self._detect_project_type()
        if project_type == "PHP":
            # Find all Python test files in tests/ directory
            tests_dir = os.path.join(self.output_dir, "tests")
            if os.path.exists(tests_dir):
                test_files = [f for f in os.listdir(tests_dir) if f.endswith('.py') and (f.startswith('test_') or 'test' in f.lower())]
                if test_files and not self.feature_test_passed:
                    # Tests were written but not executed - execute them now
                    print(f"\n⚠️  Test files were written but not executed. Executing them now...")
                    for test_file in test_files:
                        test_path = os.path.join("tests", test_file)
                        if not self._start_php_server():
                            return False, "Failed to start PHP server for testing"
                        
                        # Validate Python syntax first
                        syntax_check = self.tools.execute_command(f"python3 -m py_compile {test_path}", timeout=10, cwd=self.output_dir)
                        if syntax_check[2] != 0:
                            error_msg = f"Python syntax error in {test_path}:\n{syntax_check[1]}"
                            print(f"❌ {error_msg}")
                            return False, error_msg
                        
                        print(f"🧪 Executing test: {test_path}")
                        stdout, stderr, return_code = self.tools.execute_command(f"python3 {test_path}", timeout=60, cwd=self.output_dir)
                        self.test_counter += 1
                        status = "PASSED" if return_code == 0 else "FAILED"
                        print(f"🧪 Test #{self.test_counter}: {test_path} - {status}")
                        self._log_thought(f"Test #{self.test_counter}: {test_path} - {status}")
                        
                        if return_code != 0:
                            error_msg = f"Test failed: {test_path}\nSTDOUT: {stdout}\nSTDERR: {stderr}"
                            print(f"❌ Test failed. Output:\n{stderr}")
                            return False, error_msg
                        else:
                            self.feature_test_passed = True
        
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
        
        return False, last_test_error or "Tests did not pass"
    
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

