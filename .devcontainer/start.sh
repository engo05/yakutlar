#!/bin/bash

# Yakutlar App Startup Script for GitHub Codespaces
# This script starts the Gradio app in the background

echo "🚀 Starting Yakutlar Turkic Translation App..."

# Check if app is already running
if pgrep -f "python app.py" > /dev/null; then
    echo "⚠️  App is already running!"
    echo "📋 Check the PORTS panel in VS Code to access the app."
    exit 0
fi

# Start the app in the background
nohup python app.py > app.log 2>&1 &

# Wait a moment for the app to start
sleep 3

# Check if the app started successfully
if pgrep -f "python app.py" > /dev/null; then
    echo "✅ Yakutlar App started successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Go to the PORTS panel in VS Code"
    echo "2. Find the 'Yakutlar App' port (7860)"
    echo "3. Set visibility to 'Public' to share externally"
    echo "4. Click the URL to open the app"
    echo ""
    echo "🔒 Optional: Set authentication with 'export GRADIO_AUTH=\"user:pass\"' and restart"
    echo "📝 Check app.log for detailed logs"
else
    echo "❌ Failed to start the app. Check app.log for errors."
    tail -20 app.log
fi