import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re

def safe_eval(expr, x1, x2):
    expr = expr.replace('^', '**')
    allowed_names = {"x1": x1, "x2": x2, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt}
    return eval(expr, {"__builtins__": None}, allowed_names)

st.set_page_config(page_title="PSO Solver", layout="wide")
st.title("Particle Swarm Optimisation")

st.sidebar.header("1. Optimization Settings")
task_type = st.sidebar.selectbox("Goal", ["Minimize", "Maximize"])
bounds_val = st.sidebar.number_input("Bound Range (+/-)", value=5.0)
n_particles = st.sidebar.slider("Swarm Size", 10, 100, 30)
n_iterations = st.sidebar.slider("Max Iterations", 10, 100, 50)

st.sidebar.header("2. Hyperparameters")
w = st.sidebar.slider("Inertia (w)", 0.1, 1.2, 0.5)
c1 = st.sidebar.slider("Cognitive (c1)", 0.0, 4.0, 1.5)
c2 = st.sidebar.slider("Social (c2)", 0.0, 4.0, 1.5)

st.subheader("Define Your Problem")
col1, col2 = st.columns(2)

with col1:
    input_method = st.radio("Function Input Method", ["Manual Typing"])
    if input_method == "Manual Typing":
        func_str = st.text_input("Enter function (use x1 and x2):", "100*(x1 - x2)**2 + (1 - x2)**2")

if st.button("Run Optimization"):
    bounds = [-bounds_val, bounds_val]
    X = np.random.uniform(bounds[0], bounds[1], (n_particles, 2)) 
    V = np.zeros((n_particles, 2))
    
    p_best_pos = np.copy(X)
    p_best_val = np.array([safe_eval(func_str, p[0], p[1]) for p in X])
    
    if task_type == "Minimize":
        g_idx = np.argmin(p_best_val)
    else:
        g_idx = np.argmax(p_best_val)
        
    g_best_pos = p_best_pos[g_idx].copy()
    g_best_val = p_best_val[g_idx]

    plot_placeholder = st.empty()
    
    for t in range(n_iterations):
        for i in range(n_particles):
            r1, r2 = np.random.rand(2)
            
            V[i] = (w * V[i] + 
                    c1 * r1 * (p_best_pos[i] - X[i]) + 
                    c2 * r2 * (g_best_pos - X[i]))
            
            X[i] = np.clip(X[i] + V[i], bounds[0], bounds[1])
            
            current_val = safe_eval(func_str, X[i][0], X[i][1])
            
            if task_type == "Minimize":
                improved = current_val < p_best_val[i]
            else:
                improved = current_val > p_best_val[i]
                
            if improved:
                p_best_val[i] = current_val
                p_best_pos[i] = X[i].copy()
        
        if task_type == "Minimize":
            current_g_idx = np.argmin(p_best_val)
            if p_best_val[current_g_idx] < g_best_val:
                g_best_val = p_best_val[current_g_idx]
                g_best_pos = p_best_pos[current_g_idx].copy()
        else:
            current_g_idx = np.argmax(p_best_val)
            if p_best_val[current_g_idx] > g_best_val:
                g_best_val = p_best_val[current_g_idx]
                g_best_pos = p_best_pos[current_g_idx].copy()

        fig, ax = plt.subplots()
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[0], bounds[1])
        ax.scatter(X[:, 0], X[:, 1], c='red', alpha=0.6, label="Particles")
        ax.scatter(g_best_pos[0], g_best_pos[1], c='blue', marker='*', s=200, label="G-Best")
        ax.set_title(f"Iteration {t+1} | Best Value: {g_best_val:.4f}")
        ax.legend()
        plot_placeholder.pyplot(fig)
        plt.close(fig)

    st.success(f"Final {task_type} result: {g_best_val:.6f} at x1={g_best_pos[0]:.4f}, x2={g_best_pos[1]:.4f}")