"""
train_jepa_predictor.py — Train a latent space rollout predictor.

Replaces LLM-based counterfactual generation ("write a fake future response")
with a lightweight neural network that predicts:

    V_{t+1} = Predictor(V_t, A_t)

where:
    V_t = latent state [surprise, valence, velocity] (3-dim)
    A_t = action embedding (truncated to 32-dim from 384-dim)
    V_{t+1} = next latent state (3-dim)

Usage:
    python train_jepa_predictor.py [--data jepa_training_data.jsonl] [--output jepa_predictor.pkl]

Once trained, the predictor can simulate thousands of latent trajectories
in pure math (vector space) without touching the LLM — enabling fast
action selection via Expected Free Energy minimization.

Requires: numpy, scikit-learn (for initial version; can be upgraded to PyTorch)
"""

import json
import argparse
import pickle
import numpy as np
from pathlib import Path


def load_data(path: str) -> tuple:
    """Load JEPA training data from JSONL file.
    
    Returns:
        X: np.array of shape (N, 35) — [V_t(3) + A_t(32)]
        Y: np.array of shape (N, 3) — V_{t+1}
    """
    X_list, Y_list = [], []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            v_t = entry["v_t"]  # [surprise, valence, velocity]
            a_t = entry["action_vec"]  # truncated action embedding (32 dims)
            v_t1 = entry["v_t1"]  # next state
            
            # Input: concatenate state + action
            x = v_t + a_t  # 3 + 32 = 35 dims
            X_list.append(x)
            Y_list.append(v_t1)
    
    return np.array(X_list), np.array(Y_list)


class JEPAPredictor:
    """Lightweight latent space predictor.
    
    Architecture: 2-layer MLP with ReLU activation.
    Input: [V_t(3), A_t(32)] = 35 dims
    Output: V_{t+1}(3) = 3 dims
    
    For initial version, uses sklearn MLPRegressor.
    Can be upgraded to PyTorch for online learning.
    """
    
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_Y = None
        self.trained = False
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        """Train the predictor on (state+action, next_state) pairs."""
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        # Normalize inputs and outputs
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        Y_scaled = self.scaler_Y.fit_transform(Y)
        
        # Train/test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_scaled, Y_scaled, test_size=0.2, random_state=42)
        
        # 2-layer MLP
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            learning_rate="adaptive",
            learning_rate_init=0.001,
        )
        
        self.model.fit(X_train, Y_train)
        self.trained = True
        
        # Evaluate
        train_score = self.model.score(X_train, Y_train)
        test_score = self.model.score(X_test, Y_test)
        
        # Per-dimension error
        Y_pred_test = self.model.predict(X_test)
        Y_pred_unscaled = self.scaler_Y.inverse_transform(Y_pred_test)
        Y_test_unscaled = self.scaler_Y.inverse_transform(Y_test)
        
        dim_names = ["surprise", "valence", "velocity"]
        mae_per_dim = np.abs(Y_pred_unscaled - Y_test_unscaled).mean(axis=0)
        
        print(f"\nTraining Results:")
        print(f"  Train R²: {train_score:.4f}")
        print(f"  Test  R²: {test_score:.4f}")
        print(f"  Per-dimension MAE:")
        for name, mae in zip(dim_names, mae_per_dim):
            print(f"    {name}: {mae:.4f}")
        
        return train_score, test_score
    
    def predict(self, v_t: list, action_vec: list) -> list:
        """Predict next latent state given current state and action.
        
        Args:
            v_t: [surprise, valence, velocity] — current state
            action_vec: 32-dim truncated action embedding
            
        Returns:
            [surprise, valence, velocity] — predicted next state
        """
        if not self.trained:
            raise RuntimeError("Predictor not trained yet")
        
        x = np.array([v_t + action_vec])
        x_scaled = self.scaler_X.transform(x)
        y_scaled = self.model.predict(x_scaled)
        y = self.scaler_Y.inverse_transform(y_scaled)
        
        # Clamp to valid ranges
        result = y[0].tolist()
        result[0] = max(0.0, min(1.0, result[0]))  # surprise [0,1]
        result[1] = max(-1.0, min(1.0, result[1]))  # valence [-1,1]
        result[2] = max(0.0, min(1.0, result[2]))  # velocity [0,1]
        
        return result
    
    def rollout(self, v_t: list, action_vec: list, steps: int = 5) -> list:
        """Simulate multi-step trajectory in latent space.
        
        Predicts [V_{t+1}, V_{t+2}, ..., V_{t+steps}] by repeatedly
        applying the predictor with the same action.
        
        Returns: list of predicted states
        """
        trajectory = []
        state = v_t
        for _ in range(steps):
            state = self.predict(state, action_vec)
            trajectory.append(state)
        return trajectory
    
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler_X": self.scaler_X,
                "scaler_Y": self.scaler_Y,
                "trained": self.trained,
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "JEPAPredictor":
        pred = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        pred.model = data["model"]
        pred.scaler_X = data["scaler_X"]
        pred.scaler_Y = data["scaler_Y"]
        pred.trained = data["trained"]
        return pred


def main():
    parser = argparse.ArgumentParser(description="Train JEPA latent space predictor")
    parser.add_argument("--data", type=str, default="jepa_training_data.jsonl", help="Path to JSONL data")
    parser.add_argument("--output", type=str, default="jepa_predictor.pkl", help="Output pickle file")
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        print(f"Data file not found: {args.data}")
        print(f"Run the system for 50+ cycles to generate training data.")
        return
    
    X, Y = load_data(args.data)
    print(f"Loaded {len(X)} samples")
    print(f"  Input dims: {X.shape[1]} (3 state + 32 action)")
    print(f"  Output dims: {Y.shape[1]} (3 state)")
    
    if len(X) < 30:
        print(f"Need at least 30 samples. Current: {len(X)}")
        return
    
    predictor = JEPAPredictor()
    train_r2, test_r2 = predictor.train(X, Y)
    
    if test_r2 < 0.1:
        print(f"\nWARNING: Test R² = {test_r2:.4f} — predictor is not learning meaningful patterns yet.")
        print(f"This is expected with < 100 samples. Keep collecting data.")
    
    # Demo rollout
    print(f"\nDemo rollout from last training sample:")
    v_t = X[-1][:3].tolist()
    a_t = X[-1][3:].tolist()
    print(f"  Start: S={v_t[0]:.3f} V={v_t[1]:.3f} Vel={v_t[2]:.3f}")
    trajectory = predictor.rollout(v_t, a_t, steps=5)
    for i, state in enumerate(trajectory):
        print(f"  Step {i+1}: S={state[0]:.3f} V={state[1]:.3f} Vel={state[2]:.3f}")
    
    predictor.save(args.output)
    print(f"\nSaved predictor to {args.output}")
    print(f"Load in brain.py: predictor = JEPAPredictor.load('{args.output}')")


if __name__ == "__main__":
    main()
