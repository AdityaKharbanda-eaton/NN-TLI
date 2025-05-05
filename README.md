# NN-TLI
TITLE = Learning Signal Temporal Logic through Neural Network for Interpretable Classification,\
AUTHOR = Li, Danyang and Cai, Mingyu and Vasile, Cristian-Ioan and Tron, Roberto,\
BOOKTITLE = 2023 American Control Conference (ACC),\
YEAR = 2023,\
ORGANIZATION = IEEE

---

## Instructions to Run the Code

1. Open a bash terminal in VS Code.
2. Install `uv` using `pip install uv` if you haven't installed it before.
3. Visit [NN-TLI GitHub Repository (forked by Aditya Kharbanda)](https://github.com/AdityaKharbanda-eaton/NN-TLI#), fork the repository to your own GitHub account, and clone it locally.
   - Original repository link: [https://github.com/danyangl6/NN-TLI](https://github.com/danyangl6/NN-TLI)
4. Run `uv init --python 3.11` to initialize the environment.
5. Run `uv venv` to create a virtual environment.
6. Run `uv add -r requirements.txt` to install the required dependencies.
7. Execute `uv run main.py` to train the model and view the results in the terminal. You should achieve the best accuracy as `tensor(0.9925)` and the best rule as `F[60,60]x<21.93` and `G[0,13]y>25.07` on the naval dataset.
8. Execute `uv run visilize.py` to visualize the dataset and the rules.

