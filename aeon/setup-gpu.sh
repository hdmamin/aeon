# TODO: could mv to makefile or some new scripts dir
# But this is a good start.
sudo snap install astral-uv --classic
uv venv --seed
source .venv/bin/activate
uv sync --group gpu

# Create ipykernel. We store it in the cwd so it persists in lambda filesystem,
# then symlink it so it's discoverable by jupyter.
uv pip install ipykernel
uv run python -m ipykernel install --prefix "$(pwd)/.jupyter" --name=aeon
mkdir -p ~/.local/share/jupyter/kernels
ln -s "$(pwd)/.jupyter/share/jupyter/kernels/aeon" ~/.local/share/jupyter/kernels/aeon

