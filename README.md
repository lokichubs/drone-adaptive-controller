# Drone Adaptive Controller

Implements LQR and MRAC on a quadrotor drone and compares their performance with motor failure.

## Overview

This project implements and compares two control strategies for quadrotor drones:
- **LQR** (Linear Quadratic Regulator)
- **MRAC** (Model Reference Adaptive Control)

The controllers are tested under motor different percentage failure conditions to evaluate their robustness and performance. The point at which the LQR fails but the MRAC succeeds
