#!/usr/bin/env python3
"""
RG-Agent: Terminal-focused autonomous assistant using Ollama LLM with MCP integration
Specializes in system administration, file operations, SSH, and terminal commands
"""

import argparse
import json
import sys
import torch
import torch.nn as nn
import requests
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from datetime import datetime
import subprocess
import tempfile
import asyncio
import signal
import re
import time

# Note: Using direct subprocess communication instead of MCP SDK for simplicity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DesktopCommanderClient:
    """
    Client for interacting with Desktop Commander MCP server via subprocess
    """
    
    def __init__(self):
        self.is_connected = False
        
    async def connect(self):
        """Check if Desktop Commander is available"""
        try:
            # Test if npx and desktop-commander are available
            result = subprocess.run(
                ["npx", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                self.is_connected = True
                logger.debug("Desktop Commander (via npx) is available")
                return True
            else:
                logger.warning("npx not available - terminal commands disabled")
                return False
                
        except Exception as e:
            logger.error(f"Failed to check Desktop Commander availability: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        self.is_connected = False
        logger.info("Disconnected from Desktop Commander")
    
    async def execute_command(self, command: str, timeout: int = 30, shell: Optional[str] = None) -> Dict[str, Any]:
        """Execute a terminal command via subprocess"""
        if not self.is_connected:
            return {"error": "Not connected to Desktop Commander", "success": False}
        
        try:
            # Execute command directly using subprocess
            if shell:
                # Use specific shell
                result = subprocess.run(
                    [shell, "-c", command],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            else:
                # Use default shell
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            
            return {
                "command": command,
                "output": result.stdout + (result.stderr if result.stderr else ""),
                "return_code": result.returncode,
                "success": result.returncode == 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except subprocess.TimeoutExpired:
            return {
                "error": f"Command timed out after {timeout} seconds",
                "success": False,
                "command": command
            }
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {
                "error": str(e),
                "success": False,
                "command": command
            }

class OllamaClient:
    """
    Client for interacting with Ollama API
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "granite-code:3b"):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        self.use_cli = True  # Flag to use CLI by default, API as fallback
        
    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available"""
        # First, try CLI (default)
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0 and self.model in result.stdout:
                self.use_cli = True
                return True
        except Exception as e:
            logger.error(f"CLI check failed: {e}")
        
        # If CLI fails, try API fallback
        try:
            # First, check if Ollama is running
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return False
            
            # Try to use the model directly - some versions of Ollama 
            # don't properly report models in /api/tags
            test_data = {
                "model": self.model,
                "prompt": "test",
                "stream": False
            }
            
            test_response = self.session.post(
                f"{self.base_url}/api/generate",
                json=test_data,
                timeout=10
            )
            
            # If API works, use it as fallback
            if test_response.status_code == 200:
                logger.info("CLI failed, using API fallback")
                self.use_cli = False
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking Ollama API availability: {e}")
            
        return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response from the LLM"""
        if self.use_cli:
            return self._generate_cli(prompt, system_prompt)
        else:
            return self._generate_api(prompt, system_prompt)
    
    def _generate_cli(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using CLI"""
        try:
            # Prepare the full prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Use ollama run command
            result = subprocess.run(
                ["ollama", "run", self.model, full_prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"CLI Error: {result.stderr.strip()}"
                
        except subprocess.TimeoutExpired:
            return "Error: Request timed out"
        except Exception as e:
            return f"CLI Error: {str(e)}"
    
    def _generate_api(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using API"""
        try:
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            if system_prompt:
                data["system"] = system_prompt
                
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data.get("response", "No response received")
            else:
                # Try to get error details
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", f"Status {response.status_code}")
                    if "not found" in error_msg.lower():
                        return f"Model '{self.model}' not found. Available models: {self.get_available_models()}"
                    return f"Error: {error_msg}"
                except:
                    return f"Error: API returned status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def get_available_models(self) -> str:
        """Get list of available models"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    return ", ".join([model["name"] for model in models])
                else:
                    # Fallback to command line check
                    import subprocess
                    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')[1:]  # Skip header
                        return ", ".join([line.split()[0] for line in lines if line.strip()])
            return "Unable to retrieve model list"
        except Exception as e:
            return f"Error getting models: {e}"

class TerminalAssistant:
    """
    Terminal-focused assistant that specializes in system administration and terminal operations
    """
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.mcp_client = DesktopCommanderClient()
        self.max_iterations = 10  # Reduced for terminal tasks
        self.context_history = []  # Track recent commands and outputs
        self.mcp_connected = False
        
    async def initialize(self):
        """Initialize the assistant"""
        if await self.mcp_client.connect():
            self.mcp_connected = True
            logger.debug("MCP client initialized successfully")
            return True
        else:
            logger.warning("Failed to initialize MCP client")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.mcp_connected:
            await self.mcp_client.disconnect()
            self.mcp_connected = False

    def get_system_prompt(self) -> str:
        """Get system prompt for terminal operations"""
        return """You are a specialized terminal assistant focused on system administration and terminal operations.

Your expertise includes:
- SSH and remote server management
- File operations (find, grep, locate, search)
- Process management (ps, kill, top, htop)
- Network operations (ping, curl, wget, netstat)
- System monitoring (df, du, free, uptime)
- Text processing (awk, sed, grep, cut, sort)
- Archive operations (tar, zip, unzip)
- Permission management (chmod, chown, sudo)

🔒 CRITICAL SAFETY RULES - NEVER VIOLATE THESE:
1. NEVER generate commands that could delete system files: rm -rf /, rm -rf /*, rm -rf /*
2. NEVER generate commands that could format drives: mkfs, fdisk, parted
3. NEVER generate commands that could download and execute code: curl ... | sh, wget ... | sh
4. NEVER generate commands that could compromise security: chmod 777, dd if=/dev/zero, kill -9 1
5. NEVER generate commands that could modify system services: systemctl disable, systemctl mask
6. NEVER generate commands that could access sensitive files: cat /etc/passwd, cat /etc/shadow
7. If asked to do something destructive, respond with: "I cannot generate dangerous commands that could harm your system"

COMMAND ACCURACY REQUIREMENTS:
- For disk usage: Use 'df -h' (not 'du -sh *')
- For processes: Use 'ps aux' (not 'ps -aux')
- For file search by name: Use 'find . -name "*pattern*"' for files containing pattern
- For file search by content: Use 'grep -r "pattern" directory' (not find with -name)
- For network: Use standard command formats
- Always include proper flags and spacing

SPECIFIC EXAMPLES:
- "files containing [text]": Use 'grep -r [text] [directory]' (searches file contents)
- "find files with 'food' in name": Use 'find . -name "*food*"' (searches filenames)
- "find Python files": Use 'find . -name "*.py"' (searches by extension)
- "show disk usage": Use 'df -h' (shows filesystem usage)
- "show processes": Use 'ps aux' (shows all processes)

CRITICAL DISTINCTION:
- If task mentions "containing" or "with content" → use grep to search INSIDE files
- If task mentions "with X in name" or "named X" → use find to search FILENAMES
- If task mentions finding a "file type" → use find with -name "*.extension"
- If task doesn't mention a file extension nor file type → don't assume extensions

CRITICAL INSTRUCTIONS:
1. ALWAYS maintain the original task context throughout the conversation
2. When user provides feedback, incorporate it while keeping the original goal in mind
3. Be precise with commands - test what you would actually run
4. ALWAYS follow the exact response format below
5. SAFETY FIRST - refuse dangerous operations explicitly

REQUIRED RESPONSE FORMAT:
ACTION: [brief description of what you're doing]
COMMAND: [exact terminal command to execute]
EXPLANATION: [detailed explanation of what this command does and why it's appropriate]

EXAMPLE:
ACTION: Search for files by name pattern
COMMAND: find . -name "*pattern*" -type f
EXPLANATION: Uses find to search for files matching the specified pattern in the current directory and subdirectories.

IMPORTANT RULES:
1. Provide exactly ONE command per response
2. Make commands executable and correct
3. Include all necessary flags and parameters
4. When incorporating feedback, address the specific concerns while maintaining the original goal
5. If the original task was to find "food" files, don't change it to "foo" unless explicitly corrected
6. Be consistent with the user's requirements throughout the conversation

Current working directory: {cwd}
Current user: {user}

Remember: The user's original task is the primary goal. Feedback is meant to refine HOW you accomplish that goal, not change WHAT the goal is."""

    def is_dangerous_command(self, command: str) -> tuple[bool, str]:
        """Check if a command is dangerous"""
        dangerous_patterns = [
            (r'\brm\s+-rf\s+/', "Dangerous: recursive deletion of root directory"),
            (r'\brm\s+-rf\s+/\*', "Dangerous: recursive deletion of root files"),
            (r'\brm\s+-rf\s+\*', "Dangerous: recursive deletion of all files"),
            (r'\bmkfs\b', "Dangerous: filesystem formatting"),
            (r'\bfdisk\b', "Dangerous: disk partitioning"),
            (r'\bparted\b', "Dangerous: disk partitioning"),
            (r'\bcurl\s+.*\|\s*sh\b', "Dangerous: downloading and executing code"),
            (r'\bwget\s+.*\|\s*sh\b', "Dangerous: downloading and executing code"),
            (r'\bchmod\s+777\b', "Dangerous: overly permissive permissions"),
            (r'\bdd\s+if=', "Dangerous: disk operations"),
            (r'\bkill\s+-9\s+1\b', "Dangerous: killing init process"),
            (r'\bsystemctl\s+disable\b', "Dangerous: disabling system services"),
            (r'\bsystemctl\s+mask\b', "Dangerous: masking system services"),
            (r'\bcat\s+/etc/passwd\b', "Dangerous: accessing sensitive files"),
            (r'\bcat\s+/etc/shadow\b', "Dangerous: accessing sensitive files"),
        ]
        
        for pattern, reason in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return True, reason
        
        return False, ""

    def parse_response(self, response: str) -> Dict[str, str]:
        """Parse LLM response for terminal commands"""
        result = {}
        
        # Clean up the response
        response = response.strip()
        
        # Look for ACTION, COMMAND, EXPLANATION in structured format
        patterns = {
            'ACTION': r'ACTION:\s*(.+?)(?=\n\s*(?:COMMAND|EXPLANATION)|$)',
            'COMMAND': r'COMMAND:\s*(.+?)(?=\n\s*(?:ACTION|EXPLANATION)|$)',
            'EXPLANATION': r'EXPLANATION:\s*(.+?)(?=\n\s*(?:ACTION|COMMAND)|$)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()
        
        # If we found a structured response, check for safety
        if result.get('COMMAND'):
            is_dangerous, reason = self.is_dangerous_command(result['COMMAND'])
            if is_dangerous:
                # Override with safety refusal
                result['ACTION'] = "Refuse dangerous operation"
                result['COMMAND'] = "# Command blocked for safety"
                result['EXPLANATION'] = f"I cannot generate dangerous commands that could harm your system. {reason}"
            return result
        
        # If no structured format found, try to extract command from various formats
        command_patterns = [
            # Commands in backticks
            r'`([^`]+)`',
            # Commands after $ prompt
            r'\$\s*([^\n]+)',
            # Commands after "Command:" or similar
            r'(?:Command|command):\s*([^\n]+)',
            # Common terminal commands at start of line
            r'(?:^|\n)\s*((?:find|grep|ls|cd|mkdir|rm|cp|mv|chmod|chown|ps|kill|top|htop|df|du|free|ping|curl|wget|netstat|tar|zip|unzip|cat|tail|head|less|more|nano|vi|vim|ssh|scp|rsync|awk|sed|sort|uniq|cut|tr|wc)\s+[^\n]+)',
            # Any line that looks like a command (starts with known command)
            r'(?:^|\n)([a-zA-Z_][a-zA-Z0-9_-]*(?:\s+[^\n]+)?)',
        ]
        
        for pattern in command_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                command = match.strip()
                # Skip very short commands or obvious non-commands
                if len(command) < 2 or command.lower() in ['the', 'and', 'or', 'to', 'of', 'in', 'is', 'this', 'that', 'will', 'be', 'for', 'with', 'on', 'at', 'by', 'from']:
                    continue
                
                # Check for safety before returning
                is_dangerous, reason = self.is_dangerous_command(command)
                if is_dangerous:
                    result['ACTION'] = "Refuse dangerous operation"
                    result['COMMAND'] = "# Command blocked for safety"
                    result['EXPLANATION'] = f"I cannot generate dangerous commands that could harm your system. {reason}"
                else:
                    result['COMMAND'] = command
                    result['ACTION'] = "Execute terminal command"
                    result['EXPLANATION'] = "Command extracted from LLM response"
                return result
        
        # Last resort: try to find anything that looks like a command
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                # Check if line contains common command patterns
                if any(cmd in line.lower() for cmd in ['find', 'grep', 'ls', 'cd', 'mkdir', 'rm', 'cp', 'mv', 'chmod', 'chown', 'ps', 'kill', 'ssh', 'scp', 'tar', 'zip']):
                    # Check for safety before returning
                    is_dangerous, reason = self.is_dangerous_command(line)
                    if is_dangerous:
                        result['ACTION'] = "Refuse dangerous operation"
                        result['COMMAND'] = "# Command blocked for safety"
                        result['EXPLANATION'] = f"I cannot generate dangerous commands that could harm your system. {reason}"
                    else:
                        result['COMMAND'] = line
                        result['ACTION'] = "Execute terminal command"
                        result['EXPLANATION'] = "Command extracted from response"
                    return result
        
        return result

    async def execute_terminal_command(self, command: str, action: str = None, explanation: str = None) -> Dict[str, Any]:
        """Execute a terminal command with user authorization or feedback"""
        if not self.mcp_connected:
            return {"error": "MCP not connected", "success": False}
        
        # Display proposed command first
        print(f"\n📋 Proposed command: {command}")
        
        # Add gap
        print()
        
        # Display LLM analysis
        print(f"🤔 LLM Analysis:")
        if action:
            print(f"   🎯 What: {action}")
        if explanation:
            print(f"   💡 Why: {explanation}")
        
        # Get user authorization or feedback
        try:
            response = input("\n🔒 Execute this command? (y to execute, anything else for feedback): ").strip()
            if response.lower() == 'y':
                # Execute the command
                print(f"🔄 Executing: {command}")
                
                result = await self.mcp_client.execute_command(command)
                
                if result["success"]:
                    output = result.get('output', '')
                    print(f"✅ Success")
                    if output:
                        # Show output with proper formatting
                        print(f"📄 Output:\n{output}")
                    else:
                        print("📄 Command completed successfully (no output)")
                else:
                    print(f"❌ Error: {result.get('error', 'Command failed')}")
                
                return result
            else:
                # User provided feedback
                if response:
                    print(f"📝 Feedback received: {response}")
                    return {
                        "feedback": response,
                        "success": False,
                        "needs_retry": True,
                        "command": command
                    }
                else:
                    print("❌ No feedback provided, cancelling")
                    return {
                        "error": "No feedback provided", 
                        "success": False,
                        "cancelled": True,
                        "command": command
                    }
        except (KeyboardInterrupt, EOFError):
            print("\n❌ Command execution cancelled")
            return {
                "error": "Command cancelled by user", 
                "success": False,
                "cancelled": True,
                "command": command
            }
            
        except Exception as e:
            error_msg = f"Command execution error: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg, "success": False}

    async def handle_terminal_task(self, task: str) -> Dict[str, Any]:
        """Handle a terminal-focused task with feedback loop"""
        print(f"\n🎯 Terminal Task: {task}")
        print("=" * 60)
        
        if not self.ollama_client.is_available():
            return {"error": "Ollama is not available", "success": False}
        
        if not self.mcp_connected:
            return {"error": "MCP not connected", "success": False}
        
        # Get current context
        cwd = os.getcwd()
        user = os.getenv('USER', 'unknown')
        
        # Build context from recent history
        context = ""
        if self.context_history:
            context = f"\n\nRecent commands:\n" + "\n".join(self.context_history[-5:])
        
        # Create system prompt and base prompt
        system_prompt = self.get_system_prompt().format(cwd=cwd, user=user)
        base_prompt = f"Original task: {task}{context}"
        
        iteration = 0
        feedback_history = []
        
        try:
            while iteration < self.max_iterations:
                iteration += 1
                print(f"\n📋 Step {iteration}")
                print("-" * 30)
                
                # Build current prompt with feedback context
                if feedback_history:
                    feedback_context = "\n\nPrevious attempts and user feedback:\n" + "\n".join(feedback_history)
                    current_prompt = f"{base_prompt}{feedback_context}\n\nBased on the original task and user feedback above, provide a revised command that addresses the feedback while accomplishing the original goal."
                else:
                    current_prompt = f"{base_prompt}\n\nProvide the appropriate terminal command to accomplish this task."
                
                print("🧠 Analyzing task...")
                llm_response = self.ollama_client.generate(current_prompt, system_prompt)
                
                # Parse the response
                parsed = self.parse_response(llm_response)
                
                if not parsed.get('COMMAND'):
                    print("⚠️ No command found in response")
                    print(f"LLM Response: {llm_response}")
                    break
                
                # Execute the command (or get feedback)
                command = parsed['COMMAND']
                result = await self.execute_terminal_command(command, parsed.get('ACTION'), parsed.get('EXPLANATION'))
                
                # Handle different result types
                if result.get('needs_retry'):
                    # User provided feedback, incorporate it and try again
                    feedback = result.get('feedback', '')
                    feedback_history.append(f"Attempt {iteration}: {command}")
                    feedback_history.append(f"User feedback: {feedback}")
                    print(f"🔄 Incorporating feedback and trying again...")
                    continue
                
                elif result.get('cancelled'):
                    print(f"\n🚫 Task cancelled by user")
                    return {
                        "success": False,
                        "result": "Task cancelled by user",
                        "cancelled": True,
                        "command": command
                    }
                
                elif result["success"]:
                    # Command executed successfully
                    output = result.get('output', '')
                    
                    # Add to context history
                    self.context_history.append(f"$ {command}")
                    if output:
                        # Truncate long output for context
                        if len(output) > 200:
                            output = output[:200] + "...[truncated]"
                        self.context_history.append(f"Output: {output}")
                    
                    print(f"\n✅ Task completed successfully!")
                    if feedback_history:
                        print(f"📝 Resolved after incorporating {len(feedback_history)//2} rounds of feedback")
                    
                    return {
                        "success": True,
                        "result": f"Executed: {command}",
                        "output": result.get('output', ''),
                        "command": command,
                        "feedback_rounds": len(feedback_history)//2
                    }
                
                else:
                    # Command failed - show detailed error
                    error_msg = result.get('error', 'Command failed')
                    output = result.get('output', '')
                    return_code = result.get('return_code', 'unknown')
                    
                    self.context_history.append(f"$ {command}")
                    self.context_history.append(f"Error: {error_msg}")
                    
                    print(f"❌ Command failed:")
                    print(f"   Return code: {return_code}")
                    print(f"   Error: {error_msg}")
                    if output:
                        print(f"   Output: {output}")
                    
                    # Add failure context for next iteration
                    feedback_history.append(f"Attempt {iteration}: {command}")
                    feedback_history.append(f"System error (code {return_code}): {error_msg}")
                    if output:
                        feedback_history.append(f"Command output: {output}")
                    
                    if iteration < self.max_iterations:
                        print("🔄 Command failed, trying alternative approach...")
                        continue
                    else:
                        break
                
                # Brief pause between iterations
                await asyncio.sleep(0.5)
            
            return {
                "success": False,
                "result": f"Task could not be completed after {iteration} attempts",
                "error": "Maximum iterations reached",
                "feedback_rounds": len(feedback_history)//2
            }
            
        except Exception as e:
            error_msg = f"Task handling error: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg, "success": False}

async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="RG-Agent: Terminal-focused assistant for system administration")
    parser.add_argument("task", nargs="*", help="Terminal task to accomplish")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Start interactive mode")
    parser.add_argument("--model", default="granite-code:3b", 
                       help="Ollama model to use")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize assistant
    assistant = TerminalAssistant()
    assistant.ollama_client.model = args.model
    
    # Initialize MCP
    if args.verbose:
        print("🔧 Initializing terminal assistant...")
    if not await assistant.initialize():
        print("❌ Failed to initialize MCP connection")
        return
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print("\n\n🛑 Received shutdown signal, cleaning up...")
        asyncio.create_task(assistant.cleanup())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.interactive:
            # Interactive mode
            print("\n🖥️ RG-Agent: Terminal Assistant")
            print("Specialized in system administration and terminal operations")
            print("Type 'exit' to quit, 'help' for examples")
            print("=" * 60)
            
            while True:
                try:
                    task = input("\n🎯 What terminal task would you like me to help with? ").strip()
                    
                    if task.lower() in ["exit", "quit"]:
                        print("👋 Goodbye!")
                        break
                    
                    if task.lower() == "help":
                        print("\n📚 Example tasks:")
                        print("• 'ssh into my server at 192.168.1.100 as user admin'")
                        print("• 'find all Python files in the current directory'")
                        print("• 'search for files containing TODO in the src folder'")
                        print("• 'show me the processes using port 8080'")
                        print("• 'check disk usage of all mounted filesystems'")
                        print("• 'compress the logs folder into a tar.gz file'")
                        print("• 'find large files over 100MB in the home directory'")
                        print("\n🔄 Interactive Feedback System:")
                        print("• Type 'y' to execute proposed commands")
                        print("• Type anything else to provide feedback and try again")
                        print("• Examples: 'use a different directory', 'I meant recursive search', 'try with sudo'")
                        print("• The system will incorporate your feedback and propose a new command")
                        continue
                    
                    if not task:
                        continue
                    
                    # Handle the terminal task
                    result = await assistant.handle_terminal_task(task)
                    
                    if result["success"]:
                        feedback_rounds = result.get('feedback_rounds', 0)
                        if feedback_rounds > 0:
                            print(f"\n🎉 Task completed: {result['result']} (after {feedback_rounds} feedback rounds)")
                        else:
                            print(f"\n🎉 Task completed: {result['result']}")
                    elif result.get("cancelled"):
                        print(f"\n🚫 Task cancelled: {result['result']}")
                    else:
                        feedback_rounds = result.get('feedback_rounds', 0)
                        if feedback_rounds > 0:
                            print(f"\n😞 Task failed: {result['error']} (after {feedback_rounds} feedback rounds)")
                        else:
                            print(f"\n😞 Task failed: {result['error']}")
                    
                    print("\n" + "=" * 60)
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Unexpected error: {e}")
        
        else:
            # Single task mode
            if not args.task:
                print("Please provide a terminal task to accomplish:")
                print("Examples:")
                print("  python agent.py \"find all Python files in current directory\"")
                print("  python agent.py \"ssh into server 192.168.1.100\"")
                print("  python agent.py \"search for files containing TODO\"")
                print("  python agent.py --interactive")
                return
            
            task = " ".join(args.task)
            result = await assistant.handle_terminal_task(task)
            
            if result["success"]:
                feedback_rounds = result.get('feedback_rounds', 0)
                if feedback_rounds > 0:
                    print(f"\n🎉 Task completed: {result['result']} (after {feedback_rounds} feedback rounds)")
                else:
                    print(f"\n🎉 Task completed: {result['result']}")
            elif result.get("cancelled"):
                print(f"\n🚫 Task cancelled: {result['result']}")
                sys.exit(2)  # Different exit code for cancellation
            else:
                feedback_rounds = result.get('feedback_rounds', 0)
                if feedback_rounds > 0:
                    print(f"\n😞 Task failed: {result['error']} (after {feedback_rounds} feedback rounds)")
                else:
                    print(f"\n😞 Task failed: {result['error']}")
                sys.exit(1)
                
    finally:
        # Cleanup
        await assistant.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 