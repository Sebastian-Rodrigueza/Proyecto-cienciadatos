# Declaración de uso de inteligencia artificial

## Herramienta

Claude Code (Anthropic), usado como asistente desde la terminal y VS Code.

## Partes en las que se usó

| Parte | Uso de IA | Revisión del equipo |
|---|---|---|
| Implementación | Generación del notebook `explorar_camels_col.ipynb` y de los scripts `eda_fase1.py` y `mockup_producto.py`; configuración del entorno (`environment.yml`). | El equipo ejecutó los scripts y verificó que los resultados coinciden con el reporte solicitado y ajustó elementos que ya se encontraban definidos, como librerías incluídas desde el inicio del ejercicio. |
| Entendimiento de los datos | Identificación de problemas de formato y calidad (separadores, codificación, vacíos de fechas, columna duplicada, atípicos) y redacción del diccionario de datos. | El equipo revisó los hallazgos contra los archivos originales, confirmando que el resultado obtenido estaba acorde a lo sugerido. |
| Interpretación de resultados | Propuesta de las pruebas estadísticas (t de Welch, chi-cuadrado, Spearman, PCA) y de su interpretación. | El equipo validó que las técnicas correspondan a las vistas en clase y a los ejercicios sugeridos con las librerías abordadas previamente. |
| Documentación | Redacción y edición del reporte de la primera entrega y de este repositorio (README). | El equipo revisó, ajustó y aprobó el texto final. |
| Verificación de fuentes | Búsqueda y contraste de cifras del contexto con sus fuentes primarias (CEPAL, ENA 2022, XM) y de las licencias de los datos. | El equipo revisó las referencias citadas. |

## Prompts más relevantes

- "Con los datos adjuntos del repositorio, crea un código para entender qué datos hay en el dataset, basado en la explicación disponible en la documentación, principalmente el archivo 00_CAMELS-COL  Description (1), para exploración inicial."
- "Haz un diccionario de palabras para entender cada variable identificada en el ejercicio."
- "En un rol de ingeniero, revisa la propuesta del proyecto y para el cumplimiento de los requerimientos resume ¿qué cambiarías?"
- "Define las hipótesis de negocio que se podrían abordar desde este ejercicio, de forma medible y con técnicas vistas en el curso, de acuerdo al contenido compartido de presentaciones, con el enfoque en ciencia de datos que se requiere hasta esta etapa del curso."
- "Verifica que todo el documento adjunto como 'proyecto fase 1' esté a nivel profesional y que las cifras que puedan ser medidas coincidan con sus fuentes."
- "Ajusta parte del contenido de contexto en el documento para obtener un documento de 6 páginas, Arial 12, con lo que pide la primera entrega."

## Flujo de trabajo

1. El equipo definió el problema, la organización y los objetivos.
2. La IA generó código de exploración y análisis, que el equipo ejecutó en el entorno `proyecto`.
3. La IA propuso textos y tablas a partir de los resultados; el equipo los revisó y ajustó.
4. Todas las cifras del reporte provienen de la ejecución de los scripts del repositorio o de fuentes citadas.
