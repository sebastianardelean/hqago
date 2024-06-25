# Hybrid quantum search with genetic algorithm optimization
[![DOI](https://zenodo.org/badge/500557191.svg)](https://zenodo.org/doi/10.5281/zenodo.12535318)

## Required tools

* Anaconda
* pip
* lapack
* numpy
* networkx
* qiskit - latest version


## Environment setup (Windows & Linux)

* Install **Anaconda**
* Create a minimal environment with **Python version 3.9.7** using `conda create -n ENV_NAME python=3.9.7`
* Activate the environment using `conda activate ENV_NAME`
* Install **pip** using `conda install -c anaconda pip`
* Install **lapack** using `conda install -c conda-forge lapack`
* Install **numpy** using `pip install numpy`
* Install **networkx** using `pip install networkx`
* Install **qiskit** using `pip install qiskit`. Documentation is available [HERE](https://qiskit.org/documentation/getting_started.html)


## IBM-Q Account

* Create an account on [https://quantum-computing.ibm.com/](https://quantum-computing.ibm.com/)

## Access systems with IBM-Q account 

* Activate the newly created environment and start python REPL (type `python` in command line).
* In [https://quantum-computing.ibm.com/](https://quantum-computing.ibm.com/) go to `Account settings` and copy the the `API Token`
* In command line, enter the following code:

```python
from qiskit import IBMQ
IBMQ.save_account("TOKEN")
```
More information is available [HERE](https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq)

## Running the simulations

* Execute the Python script `qgamho_knapsack_grover_improved.py`
