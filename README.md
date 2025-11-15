# Risoluzione del Problema di Localizzazione non Capacitato (UFL)

Questo repository contiene un progetto universitario per il corso di **Algoritmi e Modelli per l'Ottimizzazione Discreta A.A.2024/2025**. L'obiettivo è implementare, confrontare e analizzare diversi algoritmi per risolvere il Problema di Localizzazione non Capacitato (Uncapacitated Facility Location - UFL).

Il progetto confronta metodi esatti (basati sulla Programmazione Lineare Intera), un'euristica classica della letteratura (l'algoritmo di Erlenkotter) e un'euristica semplice di tipo Greedy.

## Funzionalità Implementate

-   **Modelli di PLI**:
    -   Formulazione **Forte** (`y_ij <= x_i`).
    -   Formulazione **Debole** (`sum(y_ij) <= M*x_i`).
    -   Risoluzione dei **rilassamenti lineari** di entrambe le formulazioni per calcolare i lower bound.
-   **Algoritmo di Erlenkotter**:
    -   Implementazione della procedura di **Ascesa Duale** per calcolare un lower bound di alta qualità.
-   **Euristica Greedy**:
    -   Un'euristica "ADD" semplice e veloce per calcolare un upper bound di buona qualità.
-   **Generatore di Istanze**:
    -   Uno script Python per generare istanze di test di dimensioni personalizzabili, in un formato compatibile con la OR-Library.
-   **Analisi dei Risultati**:
    -   Generazione automatica di report testuali dettagliati per ogni istanza.
    -   Creazione di un file CSV riassuntivo con tutte le metriche di performance.
    -   Generazione di grafici per l'analisi visiva della qualità, del tempo di esecuzione e della scalabilità.
