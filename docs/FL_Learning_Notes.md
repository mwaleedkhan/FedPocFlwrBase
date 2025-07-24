# Federated Learning Fundamentals for Your POC

## Core Concepts You Need to Understand

### 1. What is Federated Learning?
Traditional ML: All data goes to one place → train model → deploy
Federated ML: Model goes to data → train locally → share updates → aggregate

**Key Insight**: We never share raw data, only model updates (gradients/weights)

### 2. Why Heterogeneous FL is Hard
- **System Heterogeneity**: Different devices (GPU vs neuromorphic chip)
- **Statistical Heterogeneity**: Different data distributions per device
- **Model Heterogeneity**: Different model architectures/precisions

### 3. Your Specific Challenge: Mixed Precision
```
Jetson (GPU):     32-bit floats → high precision, standard training
Akida (Neuro):    4-bit weights → low precision, spike-based learning

Problem: How do you average 32-bit and 4-bit numbers meaningfully?
```

### 4. Aggregation Strategies
**FedAvg (Standard)**:
```
new_weights = (w1*n1 + w2*n2 + ...) / (n1 + n2 + ...)
```
Where w = weights, n = number of local samples

**Your FedMPQ (Mixed Precision Quantized)**:
```
1. Dequantize 4-bit Akida weights → 32-bit
2. Average all 32-bit weights (FedAvg)  
3. Re-quantize result → send 4-bit to Akida, 32-bit to Jetson
```

### 5. Centralized vs Decentralized
**Centralized (Flower)**:
```
Server ← Client1, Client2, Client3 (star topology)
Server aggregates and broadcasts back
```

**Decentralized (ZeroMQ Gossip)**:
```
Client1 ↔ Client2 ↔ Client3 (mesh topology)
Clients average with neighbors directly
```

### 6. Your Research Questions Explained

**Q1**: Network failures happen. When a device rejoins, what happens to model quality?
**Q2**: Should disconnected devices keep learning (and risk diverging) or stay idle?
**Q3**: When rejoining, use old state or catch up with buffered updates?
**Q4**: Do these effects change as the model gets better over time?

### 7. Key Metrics You'll Track
- **Accuracy**: How good is the model?
- **Loss**: How far off are predictions?
- **Convergence**: Is the model getting better over rounds?
- **Fairness**: Are all devices contributing equally?

## Next Steps for You
1. Read through this guide
2. Ask questions about anything unclear
3. I'll build the code while you learn the concepts
4. You'll run experiments and interpret results

## Common Beginner Questions

**Q: How do I know if my FL is working?**
A: Compare to centralized baseline. FL should get 90%+ of centralized accuracy.

**Q: What if one device type dominates?**
A: We'll implement fairness constraints to ensure balanced contributions.

**Q: How do I handle device failures?**
A: That's exactly what your Q1-Q4 research questions will answer!

**Q: Is this actually useful in real life?**
A: Yes! Think privacy-preserving mobile AI, edge computing, IoT networks.