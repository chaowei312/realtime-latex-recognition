#!/usr/bin/env python
"""
Launch the Training Dashboard.

Usage:
    python run_dashboard.py
    
Then open http://localhost:5000 in your browser.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Check dependencies
    try:
        import flask
        import flask_socketio
    except ImportError:
        print("Installing dashboard dependencies...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'flask', 'flask-socketio', 'eventlet'
        ], check=True)
    
    # Run the dashboard
    dashboard_dir = Path(__file__).parent / 'training_dashboard'
    app_path = dashboard_dir / 'app.py'
    
    print("=" * 60)
    print("Starting Training Dashboard...")
    print("=" * 60)
    print()
    print("Open http://localhost:5000 in your browser")
    print()
    print("=" * 60)
    
    subprocess.run([sys.executable, str(app_path)], cwd=str(Path(__file__).parent))

if __name__ == '__main__':
    main()

