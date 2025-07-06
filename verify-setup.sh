#!/bin/bash
# Post-reboot verification script for rg-agent

echo "🔍 Verifying RG-Agent Setup..."
echo "================================"

# Check if global command exists
if command -v rg >/dev/null 2>&1; then
    echo "✅ Global 'rg' command found"
else
    echo "❌ Global 'rg' command NOT found"
    echo "   Run: sudo ln -s $(pwd)/rg-agent /usr/local/bin/rg"
    exit 1
fi

# Check if symlink points to correct location
LINK_TARGET=$(readlink /usr/local/bin/rg)
if [[ "$LINK_TARGET" == *"rg-agent"* ]]; then
    echo "✅ Symlink points to correct location: $LINK_TARGET"
else
    echo "❌ Symlink issue: $LINK_TARGET"
fi

# Check if Ollama is running
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is NOT running"
    echo "   Run: ollama serve"
    echo "   Or set up auto-start (see README)"
fi

# Check if the model is available
if ollama list 2>/dev/null | grep -q "granite-code:3b"; then
    echo "✅ granite-code:3b model is available"
else
    echo "❌ granite-code:3b model NOT found"
    echo "   Run: ollama pull granite-code:3b"
fi

# Check Node.js/npm for MCP
if command -v npx >/dev/null 2>&1; then
    echo "✅ Node.js/npx is available"
else
    echo "❌ Node.js/npx NOT found"
    echo "   Install Node.js from https://nodejs.org"
fi

# Test the global command
echo ""
echo "🧪 Testing global command..."
if rg --help >/dev/null 2>&1; then
    echo "✅ Global 'rg' command works!"
else
    echo "❌ Global 'rg' command failed"
fi

echo ""
echo "🎯 Quick test:"
echo "   Try: rg 'list files in current directory'"
echo "   Or:  rg --interactive"
