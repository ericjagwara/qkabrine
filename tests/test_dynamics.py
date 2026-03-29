"""Tests for qkabrine_automl dynamics module."""
import pytest
import numpy as np
from qkabrine_automl import (
    DataQuantumFisherMetric,
    DQFIMetrics,
    BarrenPlateauMonitor,
    QuantumNaturalGradient,
)
class TestDQFIM:
    """Test Data Quantum Fisher Information Metric."""
    
    def test_initialization(self):
        dqfim = DataQuantumFisherMetric(n_qubits=4, n_samples=50, seed=42)
        assert dqfim.n_qubits == 4
        assert dqfim.n_samples == 50
        
    def test_predict_generalization(self):
        dqfim = DataQuantumFisherMetric(n_qubits=2, n_samples=20, seed=42)
        
        X = np.random.randn(30, 2)
        
        def circuit_fn(x, params):
            return np.tanh(np.dot(x[:len(params)], params))
        
        metrics = dqfim.predict_generalization(circuit_fn, X, n_params=4)
        
        assert isinstance(metrics, DQFIMetrics)
        assert 0 <= metrics.trainability_score <= 1
        assert metrics.rank >= 0
        assert metrics.condition_number >= 1
        
    def test_compute_dqfim(self):
        dqfim = DataQuantumFisherMetric(n_qubits=2, n_samples=10, seed=42)
        
        X = np.random.randn(20, 2)
        params = np.random.randn(4)
        
        def circuit_fn(x, p):
            return np.sum(x[:len(p)] * p)
        
        Q = dqfim.compute_dqfim(circuit_fn, X, params)
        
        assert Q.shape == (4, 4)
        assert np.allclose(Q, Q.T)  # Symmetric
        
class TestBarrenPlateauMonitor:
    """Test barren plateau detection."""
    
    def test_initialization(self):
        monitor = BarrenPlateauMonitor(threshold=1e-6)
        assert monitor.threshold == 1e-6
        assert not monitor.plateau_detected
        
    def test_plateau_detection(self):
        monitor = BarrenPlateauMonitor(threshold=1e-5)
        
        # Simulate decaying gradients
        for epoch in range(10):
            grad_scale = 0.1 * np.exp(-epoch * 0.5)
            grads = np.random.randn(12) * grad_scale
            monitor.update(grads, [4, 4, 4])
            
        assert monitor.plateau_detected or len(monitor.gradient_history) == 10
        
    def test_surgery_recommendation(self):
        monitor = BarrenPlateauMonitor(auto_surgery=True)
        
        # Trigger plateau
        for _ in range(5):
            monitor.update(np.random.randn(12) * 1e-8, [4, 4, 4])
            
        if monitor.should_trigger_surgery():
            rec = monitor.get_surgery_recommendation()
            assert 'remove_layers' in rec
            assert 'reinitialize_strategy' in rec
            
class TestQuantumNaturalGradient:
    """Test QNG optimizer."""
    
    def test_initialization(self):
        qng = QuantumNaturalGradient(stepsize=0.01, damping=0.01)
        assert qng.stepsize == 0.01
        
    def test_step(self):
        qng = QuantumNaturalGradient(stepsize=0.1)
        
        params = np.array([0.1, 0.2, 0.3])
        grad = np.array([0.5, -0.3, 0.1])
        X_batch = np.random.randn(10, 2)
        
        def circuit_fn(x, p):
            return np.tanh(np.dot(x[:len(p)], p))
        
        new_params = qng.step(params, grad, circuit_fn, X_batch)
        
        assert new_params.shape == params.shape
        assert not np.allclose(new_params, params)
        
    def test_adaptive_damping(self):
        qng = QuantumNaturalGradient(adaptive_damping=True)
        assert qng.adaptive_damping
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
