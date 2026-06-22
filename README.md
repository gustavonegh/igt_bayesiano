# Estimación Bayesiana: Proyecto IGT

Este repositorio contiene la implementación computacional del trabajo final para la cátedra de Estimación Bayesiana (2026). El proyecto utiliza Inferencia Variacional (ADVI) para estimar la tasa de adopción de una nueva herramienta tecnológica en la Función Judicial de La Rioja.

## Estructura del Proyecto

* `src/inferencia_igt.py`: Script principal con el modelo bayesiano desarrollado en PyMC.
* `index.html`: Panel interactivo para la visualización directiva de los escenarios (Dashboard web).
* `docs/`: Documentación teórica y marco analítico del proyecto.

## Requisitos Previos

Para ejecutar el modelo, es necesario contar con Python 3.10+ y crear un entorno virtual para aislar las dependencias matemáticas.

## Instalación y Ejecución

1. **Clonar el repositorio y acceder al directorio:**
```bash
   git clone https://github.com/gustavonegh/igt_bayesiano.git
   cd igt_bayesiano
```

2. **Crear y activar el entorno virtual:**
* En macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
* En Windows: `python -m venv .venv` y luego `.venv\Scripts\activate`

3. **Instalar las dependencias:**
```bash
pip install pymc matplotlib numpy
```

4. **Ejecutar las simulaciones:**
```bash
python src/inferencia_igt.py
```
*Nota: Dentro del script, puede modificar la variable `escenario_elegido` (1, 2 o 3) para procesar las distintas simulaciones de telemetría descritas en el informe.*