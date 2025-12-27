"""
Training Dashboard - Web app for controlling and monitoring training.

Features:
- Start/Pause/Terminate training
- Real-time loss curve visualization
- Per-head loss breakdown (AR, TPM, SMM)
"""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'training-dashboard-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Training state
@dataclass
class TrainingState:
    status: str = "stopped"  # stopped, running, paused
    process: Optional[subprocess.Popen] = None
    current_step: int = 0
    current_epoch: int = 0
    total_loss: float = 0.0
    ar_loss: float = 0.0
    tpm_loss: float = 0.0
    smm_loss: float = 0.0
    learning_rate: float = 0.0
    start_time: Optional[float] = None
    
    # History for plotting
    steps: List[int] = None
    total_losses: List[float] = None
    ar_losses: List[float] = None
    tpm_losses: List[float] = None
    smm_losses: List[float] = None
    
    def __post_init__(self):
        self.steps = []
        self.total_losses = []
        self.ar_losses = []
        self.tpm_losses = []
        self.smm_losses = []
    
    def to_dict(self):
        return {
            'status': self.status,
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'total_loss': self.total_loss,
            'ar_loss': self.ar_loss,
            'tpm_loss': self.tpm_loss,
            'smm_loss': self.smm_loss,
            'learning_rate': self.learning_rate,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'steps': self.steps[-500:],  # Last 500 points
            'total_losses': self.total_losses[-500:],
            'ar_losses': self.ar_losses[-500:],
            'tpm_losses': self.tpm_losses[-500:],
            'smm_losses': self.smm_losses[-500:],
        }

state = TrainingState()
output_lock = threading.Lock()


def parse_log_line(line: str) -> Optional[Dict]:
    """Parse training log line to extract metrics."""
    # Pattern: Step    100 | Epoch  0 | Loss:  4.7527 | AR: 4.1778 | TPM: 0.3785 | SMM: 0.3926 | LR: 9.94e-05
    pattern = r'Step\s+(\d+)\s*\|\s*Epoch\s+(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*AR:\s*([\d.]+)\s*\|\s*TPM:\s*([\d.]+)\s*\|\s*SMM:\s*([\d.]+)\s*\|\s*LR:\s*([\d.e+-]+)'
    match = re.search(pattern, line)
    
    if match:
        return {
            'step': int(match.group(1)),
            'epoch': int(match.group(2)),
            'total_loss': float(match.group(3)),
            'ar_loss': float(match.group(4)),
            'tpm_loss': float(match.group(5)),
            'smm_loss': float(match.group(6)),
            'lr': float(match.group(7)),
        }
    return None


def training_output_reader(process):
    """Read training output and emit updates via WebSocket."""
    global state
    
    for line in iter(process.stdout.readline, ''):
        if not line:
            break
            
        line = line.strip()
        if not line:
            continue
        
        # Parse metrics from log line
        metrics = parse_log_line(line)
        
        if metrics:
            with output_lock:
                state.current_step = metrics['step']
                state.current_epoch = metrics['epoch']
                state.total_loss = metrics['total_loss']
                state.ar_loss = metrics['ar_loss']
                state.tpm_loss = metrics['tpm_loss']
                state.smm_loss = metrics['smm_loss']
                state.learning_rate = metrics['lr']
                
                # Add to history
                state.steps.append(metrics['step'])
                state.total_losses.append(metrics['total_loss'])
                state.ar_losses.append(metrics['ar_loss'])
                state.tpm_losses.append(metrics['tpm_loss'])
                state.smm_losses.append(metrics['smm_loss'])
            
            # Emit update to all connected clients
            socketio.emit('training_update', {
                'step': metrics['step'],
                'epoch': metrics['epoch'],
                'total_loss': metrics['total_loss'],
                'ar_loss': metrics['ar_loss'],
                'tpm_loss': metrics['tpm_loss'],
                'smm_loss': metrics['smm_loss'],
                'lr': metrics['lr'],
            })
        
        # Also emit raw log line
        socketio.emit('log_line', {'line': line})
    
    # Process ended
    with output_lock:
        state.status = "stopped"
        state.process = None
    
    socketio.emit('training_stopped', {})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/state')
def get_state():
    """Get current training state."""
    with output_lock:
        return jsonify(state.to_dict())


@app.route('/api/start', methods=['POST'])
def start_training():
    """Start training process."""
    global state
    
    with output_lock:
        if state.status == "running":
            return jsonify({'error': 'Training already running'}), 400
        
        # Get training config from request
        config = request.json or {}
        batch_size = config.get('batch_size', 32)
        lr = config.get('lr', '1e-4')
        epochs = config.get('epochs', 10)
        warmup_steps = config.get('warmup_steps', 1000)
        log_interval = config.get('log_interval', 50)
        eval_interval = config.get('eval_interval', 500)
        save_interval = config.get('save_interval', 2500)
        num_workers = config.get('num_workers', 0)
        lambda_position = config.get('lambda_position', 1.0)
        lambda_stroke = config.get('lambda_stroke', 0.5)
        max_grad_norm = config.get('max_grad_norm', 1.0)
        weight_decay = config.get('weight_decay', 0.01)
        experiment_name = config.get('experiment_name', 'dashboard')
        log_dir = config.get('log_dir', 'logs/dashboard_train')
        
        # Build command
        project_root = Path(__file__).parent.parent
        cmd = [
            sys.executable, '-m', 'context_tracker.training.train',
            '--batch_size', str(batch_size),
            '--lr', str(lr),
            '--epochs', str(epochs),
            '--warmup_steps', str(warmup_steps),
            '--log_dir', str(log_dir),
            '--log_interval', str(log_interval),
            '--eval_interval', str(eval_interval),
            '--save_interval', str(save_interval),
            '--experiment_name', str(experiment_name),
            '--num_workers', str(num_workers),
        ]
        
        # Start process
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(project_root),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
            )
            
            state.process = process
            state.status = "running"
            state.start_time = time.time()
            state.steps = []
            state.total_losses = []
            state.ar_losses = []
            state.tpm_losses = []
            state.smm_losses = []
            
            # Start output reader thread
            reader_thread = threading.Thread(target=training_output_reader, args=(process,))
            reader_thread.daemon = True
            reader_thread.start()
            
            return jsonify({'status': 'started'})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/api/stop', methods=['POST'])
def stop_training():
    """Stop training process."""
    global state
    
    with output_lock:
        if state.process is None:
            return jsonify({'error': 'No training running'}), 400
        
        try:
            if os.name == 'nt':
                # Windows: send CTRL+BREAK
                state.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # Unix: send SIGTERM
                state.process.terminate()
            
            state.status = "stopping"
            return jsonify({'status': 'stopping'})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/api/kill', methods=['POST'])
def kill_training():
    """Force kill training process."""
    global state
    
    with output_lock:
        if state.process is None:
            return jsonify({'error': 'No training running'}), 400
        
        try:
            state.process.kill()
            state.status = "stopped"
            state.process = None
            return jsonify({'status': 'killed'})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    with output_lock:
        emit('training_state', state.to_dict())


if __name__ == '__main__':
    print("=" * 60)
    print("Training Dashboard")
    print("=" * 60)
    print("Open http://localhost:5000 in your browser")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

