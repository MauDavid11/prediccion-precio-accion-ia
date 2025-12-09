# 📈 Predicción del Precio de Cierre de Acciones con IA (LSTM)

Este proyecto implementa un modelo de **Red Neuronal LSTM** para predecir el **precio de cierre del día siguiente** de una acción, utilizando datos históricos descargados automáticamente desde Yahoo Finance.

El modelo usa:

- Datos **post-pandemia** (desde 2020)
- Ventana deslizante de **60 días**
- LSTM profunda con *Dropout*
- Métricas MAE y RMSE
- Predictivo sobre el **cierre de HOY**, usando únicamente el cierre de AYER

Incluye además:

- 🖥️ **CLI (Command Line Interface)** para ejecutar el modelo desde terminal  
- 🌐 **Aplicación web completa en Streamlit** para predicciones interactivas  
- 📊 **Gráficas automáticas** de predicción vs. reales  
- 🚀 **Recomendado para ejecutarse con GPU T4 en Google Colab**

---

## 📌 Contenido del repositorio
