# Heart Lung Machine – Virtual Lab (Ex. No. 3)

Interactive Streamlit simulation of the heart-lung machine (cardiopulmonary bypass) model.
Educational simulation only – not for real patient care.

## Run in VS Code
1. Install Python 3.10+ and the VS Code **Python** extension.
2. `File > Open Folder...` and choose this `hlm_lab` folder.
3. Open the terminal (`Ctrl + ~`) and run:

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
4. The browser opens at http://localhost:8501.
   (Or press F5 and choose "Run HLM Lab (Streamlit)".)

## Experiment flow
1. Operating Room – register patient, compute BSA and target flow (2.4 L/min/m²)
2. Equipment Room – verify HLM model, cables, effluent pump, blood pump
3. Circuit Assembly – connect tubing in the correct order, connect effluent pump
4. Priming & Leak Check – fill compartments, set low speed, prime, find bubbles/leaks
5. Run the Machine – power modules, heater, blood/effluent pump; observe circulation
6. Viva & Report – observation table, viva questions, downloadable report
