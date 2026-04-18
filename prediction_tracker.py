import json
import os
from datetime import datetime

HISTORY_FILE = "prediction_history.json"

def save_prediction(prediction_data):
    """
    Saves a prediction to history.
    prediction_data: dict with keys 'symbol', 'signal', 'confidence', 'price', 'timestamp', 'actual_result'
    """
    history = load_history()
    
    # Add timestamp if not present
    if 'timestamp' not in prediction_data:
        prediction_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepend to history
    history.insert(0, prediction_data)
    
    # Keep only last 100 predictions
    history = history[:100]
    
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error saving history: {e}")

def load_history():
    """Loads prediction history from JSON."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def load_accuracy():
    """
    Calculates accuracy based on stored results.
    Returns (accuracy_ratio, total_trades)
    """
    history = load_history()
    # Only count predictions that have an actual_result
    evaluated = [p for p in history if p.get('correct') is not None]
    
    if not evaluated:
        return 0.0, 0
    
    wins = sum(1 for p in evaluated if p['correct'])
    total = len(evaluated)
    
    return wins / total, total

def update_prediction_result(timestamp, actual_price):
    """
    Updates a past prediction with the actual price movement result.
    This would be called when new data arrives to verify past signals.
    """
    history = load_history()
    updated = False
    
    for pred in history:
        if pred['timestamp'] == timestamp and pred.get('correct') is None:
            # Logic to determine if prediction was correct
            # For simplicity: if signal was BUY and price went up, correct=True
            entry_price = pred.get('price', 0)
            signal = pred.get('signal', '')
            
            if entry_price > 0:
                if "BUY" in signal:
                    pred['correct'] = actual_price > entry_price
                elif "SELL" in signal:
                    pred['correct'] = actual_price < entry_price
                else:
                    pred['correct'] = None # Neutral/Hold
                updated = True
                pred['actual_price'] = actual_price
    
    if updated:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
