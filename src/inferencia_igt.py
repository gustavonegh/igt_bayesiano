"""
TRABAJO FINAL - ESTIMACIÓN BAYESIANA 2026
Equipo: Glaria (Micaela), Mercado (Gustavo), Veron (Mirta).
Caso de Estudio: Inferencia Variacional aplicada a la Adopción de la 
                 Herramienta IGT en el Portal del Poder Judicial.
"""

import pymc as pm
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. DEFINICIÓN DE LOS DATOS (SIMULADOR)
# ==========================================
# Cambiá este número (1, 2 o 3) para correr los distintos escenarios
escenario_elegido = 3  

if escenario_elegido == 1:
    titulo_escenario = "Simulación 1: Lanzamiento Base"
    visitas_totales = 2500  
    consultas_jurisprudencia = 415  
elif escenario_elegido == 2:
    titulo_escenario = "Simulación 2: Resistencia (Alta Demanda)"
    visitas_totales = 5000  
    consultas_jurisprudencia = 500
elif escenario_elegido == 3:
    titulo_escenario = "Simulación 3: Éxito (Post Capacitación)"
    visitas_totales = 3000  
    consultas_jurisprudencia = 850

print(f"\n--- {titulo_escenario} ---")
print(f"Tasa de adopción observada (frecuentista): {consultas_jurisprudencia / visitas_totales:.4f}")

# ==========================================
# 2. CONSTRUCCIÓN DEL MODELO BAYESIANO
# ==========================================
with pm.Model() as modelo_igt:
    
    # A) DISTRIBUCIÓN A PRIORI (Prior)
    # Utilizamos una distribución Beta(2, 10). 
    # Representa una creencia conservadora de que la adopción de nuevas 
    # tecnologías en el ámbito judicial lleva tiempo y no será masiva de entrada.
    theta = pm.Beta('tasa_adopcion', alpha=2, beta=10)
    
    # B) VEROSIMILITUD (Likelihood)
    # Modelamos los datos observados como una distribución Binomial.
    y_obs = pm.Binomial('y_obs', n=visitas_totales, p=theta, observed=consultas_jurisprudencia)
    
    # ==========================================
    # 3. ESTIMACIÓN COMPUTACIONAL (INFERENCIA VARIACIONAL)
    # ==========================================
    print("\nIniciando Inferencia Variacional (ADVI)...")
    
    # En lugar de usar MCMC, usamos ADVI (Automatic Differentiation Variational Inference)
    # que aproxima la distribución posterior mediante optimización.
    aproximacion = pm.fit(n=30000, method='advi', obj_optimizer=pm.adam(learning_rate=0.01))
    
    # Muestreamos 10,000 valores de la distribución aproximada para graficar
    traza_variacional = aproximacion.sample(10000)

# ==========================================
# 4. VISUALIZACIÓN Y DIAGNÓSTICO (100% Nativo con Numpy)
# ==========================================

# 4.1 Gráfico de pérdida (ELBO)
plt.figure(figsize=(8, 4))
plt.plot(aproximacion.hist, color='steelblue')
plt.ylabel('Pérdida (ELBO)')
plt.xlabel('Iteraciones')
plt.title('Convergencia de la Inferencia Variacional (ADVI)')
plt.tight_layout()
plt.show()

# 4.2 Gráfico de la Distribución Posterior
# Extraemos los valores simulados
muestras = traza_variacional.posterior['tasa_adopcion'].values.flatten()

# Calculamos la Media y el Intervalo de Credibilidad (IC) del 94% usando percentiles puros
media_posterior = np.mean(muestras)
limite_inf = np.percentile(muestras, 3)   # Límite inferior (3%)
limite_sup = np.percentile(muestras, 97)  # Límite superior (97%)

plt.figure(figsize=(8, 5))
# Graficamos el histograma de la campana
plt.hist(muestras, bins=40, density=True, color='darkorange', alpha=0.7)

# Agregamos las líneas de la Media y los límites del IC
plt.axvline(media_posterior, color='red', linestyle='dashed', linewidth=2, 
            label=f'Media: {media_posterior:.3f}')
plt.axvline(limite_inf, color='black', linestyle='dotted', linewidth=2, 
            label=f'IC 94% Inf: {limite_inf:.3f}')
plt.axvline(limite_sup, color='black', linestyle='dotted', linewidth=2, 
            label=f'IC 94% Sup: {limite_sup:.3f}')

plt.title('Distribución Posterior: Tasa de Adopción IGT-Rioja')
plt.xlabel('Tasa de Adopción')
plt.ylabel('Densidad')
plt.legend()
plt.tight_layout()
plt.show()

# 4.3 Resumen estadístico por consola manual
print("\n--- RESUMEN DE LA ESTIMACIÓN BAYESIANA ---")
print(f"Media a posteriori observada: {media_posterior:.4f} (Aprox {media_posterior*100:.1f}%)")
print(f"Intervalo de Credibilidad del 94%: [{limite_inf:.4f}, {limite_sup:.4f}]")