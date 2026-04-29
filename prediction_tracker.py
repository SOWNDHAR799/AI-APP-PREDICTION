import json
import os
from datetime import datetime

HISTORY_FILE = "prediction_history.json"

def save_prediction(prediction_data):
    """
    Saves a prediction to history.
    prediction_data: dict with keys 'symbol', 'signal', 'confidence', 'price', 'timestamp', 'catalyst'
    """
    history = load_history()
    
    # Add timestamp if not present
    if 'timestamp' not in prediction_data:
        prediction_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Ensure catalyst exists
    if 'catalyst' not in prediction_data:
        prediction_data['catalyst'] = "Technical Analysis"
        
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
            data = json.load(f)
            # Migration: Ensure all old entries have 'catalyst'
            for item in data:
                if 'catalyst' not in item:
                    item['catalyst'] = "Broad Market Trends"
            return data
    except Exception:
        return []

def load_advanced_stats():
    """
    Step 7: Professional Track Record Engine.
    Calculates detailed performance metrics.
    """
    history = load_history()
    evaluated = [p for p in history if p.get('correct') is not None]
    
    if not evaluated:
        return {
            "win_rate": 0.0, "total": 0, "avg_profit": 0.0, 
            "avg_loss": 0.0, "profit_factor": 0.0, "rr_actual": 0.0
        }
    
    wins = [p for p in evaluated if p['correct']]
    losses = [p for p in evaluated if not p['correct']]
    
    total = len(evaluated)
    win_rate = len(wins) / total
    
    # Calculate % returns based on price change
    def get_ret(p):
        entry = p.get('price', 0)
        exit = p.get('actual_price', 0)
        if entry == 0: return 0
        ret = abs(exit - entry) / entry * 100
        return ret

    avg_win = sum(get_ret(w) for w in wins) / len(wins) if wins else 0.0
    avg_loss = sum(get_ret(l) for l in losses) / len(losses) if losses else 0.0
    
    total_win_amt = sum(get_ret(w) for w in wins)
    total_loss_amt = sum(get_ret(l) for l in losses)
    profit_factor = total_win_amt / total_loss_amt if total_loss_amt > 0 else total_win_amt
    
    return {
        "win_rate": win_rate * 100,
        "total": total,
        "avg_profit": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "rr_actual": avg_win / avg_loss if avg_loss > 0 else 0.0
    }

def update_prediction_result(timestamp, actual_price):
    """
    Updates a past prediction with the actual price movement result.
    """
    history = load_history()
    updated = False
    
    for pred in history:
        if pred['timestamp'] == timestamp and pred.get('correct') is None:
            entry_price = pred.get('price', 0)
            signal = pred.get('signal', '')
            
            if entry_price > 0:
                if "BUY" in signal:
                    pred['correct'] = actual_price > entry_price
                elif "SELL" in signal:
                    pred['correct'] = actual_price < entry_price
                else:
                    pred['correct'] = None 
                
                if pred['correct'] is not None:
                    updated = True
                    pred['actual_price'] = actual_price
    
    if updated:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)

def auto_verify_signals(price_fetcher_fn):
    """
    Automatically verifies pending signals by fetching current prices.
    price_fetcher_fn: a function that takes a symbol and returns current price.
    """
    history = load_history()
    pending = [p for p in history if p.get('correct') is None and "HOLD" not in p.get('signal', '') and "NO TRADE" not in p.get('signal', '')]
    
    if not pending:
        return
        
    updated = False
    for pred in pending:
        # Check if enough time has passed (at least 30 mins) to avoid instant noise verification
        try:
            ts = datetime.strptime(pred['timestamp'], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - ts).total_seconds() < 60: # 1 min to avoid instant noise
                continue
                
            current_price = price_fetcher_fn(pred['symbol'])
            if current_price:
                entry_price = pred.get('price', 0)
                target = pred.get('target', 0)
                sl = pred.get('sl', 0)
                signal = pred.get('signal', '')
                
                # Institutional Verification (Target vs SL)
                if "BUY" in signal:
                    if target > 0 and current_price >= target:
                        pred['correct'] = True
                        pred['status'] = "TARGET HIT 🎯"
                    elif sl > 0 and current_price <= sl:
                        pred['correct'] = False
                        pred['status'] = "SL HIT 🛑"
                    elif (datetime.now() - ts).total_seconds() > 86400: # After 24h, check direction
                        pred['correct'] = current_price > entry_price
                        pred['status'] = "WIN ✅" if pred['correct'] else "LOSS ❌"
                elif "SELL" in signal:
                    if target > 0 and current_price <= target:
                        pred['correct'] = True
                        pred['status'] = "TARGET HIT 🎯"
                    elif sl > 0 and current_price >= sl:
                        pred['correct'] = False
                        pred['status'] = "SL HIT 🛑"
                    elif (datetime.now() - ts).total_seconds() > 86400: # After 24h, check direction
                        pred['correct'] = current_price < entry_price
                        pred['status'] = "WIN ✅" if pred['correct'] else "LOSS ❌"
                
                if pred.get('correct') is not None:
                    pred['actual_price'] = current_price
                    updated = True
        except:
            continue
            
    if updated:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)

# Alias for backward compatibility
load_accuracy = load_advanced_stats
