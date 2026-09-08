# Bull's AI

This repository contains all the modules needed to assemble and run the final trading strategy that picks the stocks to short/long every month.
The main file to run is `main.py` which selects the monthly stocks to short/long for the Out Of Sample period (2015-01 - 2025-06) and saves this portfolio to a csv file.
This file also runs the pairs trading strategy (implemented in `pairs_trading.py`)  and saves it to anohter csv will all the pairs that were traded across the Out of Sample Period. The pairs trading is meant to complement the main strategy, and allowed us to explore a trading strategy that was completely new to us.
`metrics.py` combines these 2 csv's into one big portfolio and calculates various Portfolio Performance Statistics for it.