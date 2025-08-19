#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 20:45:10 2025

@author: Sahar Jahani

This script compares the performance of RLS and LMS adaptive filters on a noisy 
sinusoidal signal. Both filters attempt to estimate the true underlying signal 
from noisy observations.
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate the time vector and the true sine wave
n = 300
t = np.linspace(0, 4 * np.pi, n)
true_signal = 2 * np.sin(t)

# Generate additive white Gaussian noise
noise = np.random.normal(0, 0.5, n)

# Add noise to the true signal to simulate observations
Output = true_signal + noise

# --------------------------
# RLS Adaptive Filter
# --------------------------

M = 8              # Filter order (number of taps)
lambdA = 0.99       # Forgetting factor (close to 1 = longer memory)
delta = 0.5         # Initial value for inverse correlation matrix
wrls = np.zeros(M)  # Initial filter weights
P = (1 / delta) * np.eye(M)  # Inverse correlation matrix

e = np.zeros(n)     # Error signal (difference between desired and estimated)
y = np.zeros(n)     # RLS output signal (estimated signal)
weights_rls = np.zeros((n, M))  # Store weight evolution for plotting

# Apply RLS algorithm
for i in range(M, n):
    x = Output[i-M:i][::-1]  # Most recent M input samples, reversed
    y[i] = wrls @ x          # Filter output (dot product)
    d = true_signal[i]       # Desired output at time i
    e[i] = d - y[i]          # Estimation error

    # Compute Kalman gain vector
    k = P @ x / (lambdA + x.T @ P @ x)

    # Update filter weights
    wrls = wrls + k * e[i]

    # Store weights
    weights_rls[i, :] = wrls

    # Update inverse correlation matrix
    P = (P - np.outer(k, P @ x)) / lambdA

# --------------------------
# LMS Adaptive Filter
# --------------------------

# Learning rate calculation (based on average input power)
p = np.mean(Output**2)
mu = 1 / (10 * p)  # Recommended stability criterion

wlms = np.zeros(M)     # LMS filter weights
y_current = np.zeros(n)  # LMS output signal
elms = np.zeros(n)       # LMS error signal
weights_lms = np.zeros((n, M))  # Store weight evolution

# Apply LMS algorithm
for i in range(M, n):
    x = Output[i-M:i][::-1]  # Input vector
    y_current[i] = wlms @ x  # Filter output
    elms[i] = true_signal[i] - y_current[i]  # Prediction error
    wlms = wlms + mu * elms[i] * x  # Update weights

    # Store weights
    weights_lms[i, :] = wlms

# --------------------------
# Plot Signal Estimation Results
# --------------------------

plt.figure(figsize=(12, 5))
plt.plot(t, true_signal, label='True Signal', linewidth=2)
plt.plot(t, Output, label='Noisy Input', color='gray')
plt.plot(t, y, label='RLS Output', linewidth=2)
plt.plot(t, y_current, '-.', label='LMS Output')
plt.xlabel('Time', fontsize=14, fontweight='bold')
plt.ylabel('Signal', fontsize=14, fontweight='bold')
plt.title('LMS vs RLS Adaptive Filtering', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Format legend font
legend = plt.legend()
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_fontsize(12)

plt.show()

# --------------------------
# Plot Weight Convergence
# --------------------------

plt.figure(figsize=(12, 6))
for m in range(M):
    plt.plot(weights_rls[:, m], label=f'RLS w[{m}]', linestyle='-')
    plt.plot(weights_lms[:, m], label=f'LMS w[{m}]', linestyle='--')

plt.title('Weight Convergence: LMS vs RLS', fontsize=14, fontweight='bold')
plt.xlabel('Iteration', fontsize=14, fontweight='bold')
plt.ylabel('Weight Value', fontsize=14, fontweight='bold')
plt.grid(True)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()


# --------------------------
# Plot MSE Comparison
# --------------------------


mse_rls = e**2
mse_lms = elms**2


plt.figure(figsize=(10, 4))
plt.plot(mse_rls, label='RLS MSE')
plt.plot(mse_lms, label='LMS MSE', linestyle='--')
plt.title('Mean Squared Error: LMS vs RLS', fontsize=14, fontweight='bold')
plt.xlabel('Iteration', fontsize=14, fontweight='bold')
plt.ylabel('MSE', fontsize=14, fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()