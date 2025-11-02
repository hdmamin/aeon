.PHONY nanochat

# Pull changes from karpathy's original nanochat repo.
nanochat:
	git fetch nanochat
	git subtree pull --prefix=nanochat nanochat master --squash

