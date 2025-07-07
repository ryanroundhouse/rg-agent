# RG-Agent: Terminal-Focused AI Assistant with Interactive Feedback

A specialized terminal assistant that uses local Ollama LLM (granite-code:3b) to help with system administration, file operations, SSH, and terminal commands. Features an intelligent feedback system that learns from your corrections.

## 🚀 Features

- 🖥️ **Terminal-focused expertise**: Specialized in system administration and command-line operations
- 🔄 **Interactive feedback system**: Refine commands through conversation until they're exactly right
- 🌐 **Global accessibility**: Use `rg` command from anywhere on your system
- 🛡️ **Safety first**: All commands require authorization before execution
- 🧠 **Context awareness**: Maintains task context and incorporates feedback intelligently
- 🚀 **Local Ollama integration**: Uses your local granite-code:3b model
- 🔧 **MCP integration**: Executes commands via Desktop Commander MCP server

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+ (for MCP integration)
- Ollama with granite-code:3b model

### Quick Setup

1. **Clone and setup the environment:**
```bash
git clone <your-repo-url>
cd rg-agent
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Install Desktop Commander MCP Server:**
```bash
npm install -g @wonderwhy-er/desktop-commander
```

3. **Setup Ollama:**
```bash
# Install and start Ollama
ollama pull granite-code:3b
ollama serve
```

4. **Setup global `rg` command:**
```bash
# Make the wrapper executable
chmod +x rg-agent

# Create global symlink
sudo ln -s "$(pwd)/rg-agent" /usr/local/bin/rg
```

5. **Test the installation:**
```bash
# Test locally
./rg-agent --help

# Test globally
cd ~
rg --help
```

## 🎯 Usage

### Global Command (Recommended)

Once set up, use `rg` from anywhere:

```bash
# From any directory
rg "find all Python files in current directory"
rg "ssh into my server at 192.168.1.100"
rg "check disk usage of all mounted filesystems"

# Interactive mode
rg --interactive
```

### ⚠️ Important: Always Use Quotes!

Due to shell expansion, **always wrap your commands in quotes** to avoid issues with special characters like `?`, `*`, `[`, `{`, etc.

**❌ This will cause shell expansion errors:**
```bash
rg what are the ssh keys I have on my system?
# Error: zsh: no matches found: system?
```

**✅ This works correctly:**
```bash
rg "what are the ssh keys I have on my system?"
```

The wrapper script will detect problematic characters and provide helpful guidance when possible.

### Interactive Feedback System

The key feature is the interactive feedback loop:

```bash
$ rg "find files containing TODO"

🤔 LLM Analysis:
   🎯 What: Search for files containing 'TODO' text
   💡 Why: Uses grep to search for 'TODO' in file contents

📋 Proposed command: grep -r "TODO" .
�� Execute this command? (y to execute, anything else for feedback): I meant files with TODO in the filename

📝 Feedback received: I meant files with TODO in the filename
🔄 Incorporating feedback and trying again...

🤔 LLM Analysis:
   🎯 What: Find files with 'TODO' in their filename
   💡 Why: Uses find to search for files containing 'TODO' in their names

📋 Proposed command: find . -name "*TODO*" -type f
🔒 Execute this command? (y to execute, anything else for feedback): y

✅ Task completed successfully!
```

## 🎓 Terminal Expertise

The agent specializes in:

### **System Administration**
- **SSH & Remote Management**: `ssh`, `scp`, `rsync`
- **Process Management**: `ps`, `kill`, `top`, `htop`
- **System Monitoring**: `df`, `du`, `free`, `uptime`
- **Permission Management**: `chmod`, `chown`, `sudo`

### **File Operations**
- **Search & Find**: `find`, `grep`, `locate`
- **Text Processing**: `awk`, `sed`, `cut`, `sort`
- **Archive Operations**: `tar`, `zip`, `unzip`
- **File Manipulation**: `cp`, `mv`, `rm`, `mkdir`

### **Network Operations**
- **Connectivity**: `ping`, `curl`, `wget`
- **Network Info**: `netstat`, `ss`, `lsof`
- **Troubleshooting**: Network diagnostics and monitoring

## 💡 Example Tasks

### Quick Commands
```bash
# File operations
rg "find all Python files in current directory"
rg "search for files containing TODO in src folder"
rg "find large files over 100MB"

# System monitoring
rg "show me processes using port 8080"
rg "check disk usage of all mounted filesystems"
rg "show me memory usage"

# Network operations
rg "ping google.com"
rg "check if port 22 is open on server"
rg "download file from URL"

# SSH and remote
rg "ssh into my server at 192.168.1.100 as user admin"
rg "copy file to remote server"
rg "sync local directory with remote"
```

### Interactive Mode
```bash
# Start interactive session
rg --interactive

# Then you can have conversations like:
🎯 What terminal task would you like me to help with? find python files
🤔 LLM Analysis: ...
📋 Proposed command: find . -name "*.py" -type f
🔒 Execute this command? (y to execute, anything else for feedback): only in current directory, not subdirectories

📝 Feedback received: only in current directory, not subdirectories
�� Incorporating feedback and trying again...
📋 Proposed command: find . -maxdepth 1 -name "*.py" -type f
🔒 Execute this command? (y to execute, anything else for feedback): y
✅ Task completed successfully!
```

## 🔄 Interactive Feedback System

The agent's key innovation is its feedback system:

### **How It Works**
1. **Analyze your request** using AI
2. **Propose a command** with explanation
3. **Wait for your input**:
   - Type `y` to execute
   - Type anything else to provide feedback
4. **Incorporate feedback** and try again
5. **Repeat until perfect**

### **Feedback Examples**
- `"use a different directory"`
- `"I meant recursive search"`
- `"try with sudo"`
- `"exclude hidden files"`
- `"search case-insensitively"`
- `"also check file contents"`

## 🛡️ Safety Features

- **Authorization required**: Every command needs your approval
- **Clear explanations**: See exactly what each command does
- **Feedback loop**: Correct misunderstandings before execution
- **Context preservation**: Original task never gets lost
- **Cancellation**: Easy to cancel with Ctrl+C

## 🔧 CLI Options

```bash
# Basic usage
rg "your task description"

# Interactive mode
rg --interactive
rg -i

# Different model
rg "task" --model your-model-name

# Verbose logging
rg "task" --verbose
rg "task" -v

# Help
rg --help
```

## 🏗️ Architecture

```
User Input → Terminal Assistant → {
    LLM Analysis → Command Generation → User Approval → {
        'y' → Execute Command → Success/Failure
        'feedback' → Incorporate Feedback → Retry with New Command
    }
}
```

**Core Components:**
- **TerminalAssistant**: Main orchestrator with feedback loop
- **OllamaClient**: Local LLM integration (CLI + API fallback)
- **DesktopCommanderClient**: MCP integration for safe command execution
- **Interactive Feedback System**: Intelligent command refinement

## 🔧 Troubleshooting

### Common Issues

**"Ollama is not available"**
```bash
# Start Ollama service
ollama serve

# Check if model is installed
ollama list
ollama pull granite-code:3b
```

**"MCP not connected"**
```bash
# Check Node.js installation
node --version
npm --version

# Reinstall Desktop Commander
npm install -g @wonderwhy-er/desktop-commander
```

**"Command not found: rg"**
```bash
# Check if symlink exists
ls -la /usr/local/bin/rg

# Recreate if needed
cd /path/to/rg-agent
sudo ln -s "$(pwd)/rg-agent" /usr/local/bin/rg

# Check PATH
echo $PATH | grep -o '/usr/local/bin'
```

**Virtual environment errors**
```bash
# The rg-agent script should handle this automatically
# If issues persist, check the script resolves symlinks correctly
```

## 🚀 After System Reboot

After rebooting your system, you may need to:

1. **Start Ollama (if not auto-starting):**
```bash
ollama serve
```

2. **Test the global command:**
```bash
rg --help
```

3. **Everything else should work automatically:**
   - The global `rg` symlink persists
   - Virtual environment is handled by the script
   - MCP server is globally installed

## 🎯 Tips for Best Results

### **Write Clear Requests**
- ✅ `"find all Python files in current directory"`
- ✅ `"ssh into my server at 192.168.1.100"`
- ❌ `"find files"` (too vague)

### **Use the Feedback System**
- Don't just cancel - provide feedback!
- Be specific about what you want different
- The AI learns from your corrections

### **Quote Your Commands**
```bash
# Good
rg "where are my pictures?"
rg "find files containing TODO"

# Problematic (shell interpretation)
rg where are my pictures?
rg find files containing TODO
```

## 📝 License

MIT License

---

## 🆘 Need Help?

1. **Test components individually:**
   - Ollama: `ollama list`
   - Global command: `rg --help`
   - MCP: Check if commands execute properly

2. **Use interactive mode for complex tasks:**
   ```bash
   rg --interactive
   ```

3. **Provide detailed feedback:** The more specific your feedback, the better the results

4. **Check logs:** Use `--verbose` flag for detailed logging

The agent is designed to be conversational - treat it like a skilled terminal expert who can learn from your corrections!
