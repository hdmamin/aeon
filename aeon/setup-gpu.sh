# TODO: could mv to makefile or some new scripts dir
# But this is a good start.
curl -LsSf https://astral.sh/uv/install.sh | sh
# python >=3.12 was causing unsloth errors. Create a separate env to avoid breaking cpu venv.
uv venv .venv-gpu --python 3.11 --seed
source .venv-gpu/bin/activate
uv sync --active --group gpu

# Create ipykernel. We store it in the cwd so it persists in lambda filesystem,
# then symlink it so it's discoverable by jupyter.
uv pip install ipykernel
uv run --active python -m ipykernel install --prefix "$(pwd)/.jupyter" --name=aeon
mkdir -p ~/.local/share/jupyter/kernels
ln -s "$(pwd)/.jupyter/share/jupyter/kernels/aeon" ~/.local/share/jupyter/kernels/aeon

