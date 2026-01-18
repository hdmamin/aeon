# TODO: could mv to makefile or some new scripts dir
# But this is a good start.
sudo snap install astral-uv --classic
uv venv --seed
source .venv/bin/activate
uv sync --group gpu
uv pip install ipykernel
uv run python -m ipykernel install --prefix "$(pwd)/.jupyter" --user --name=aeon
