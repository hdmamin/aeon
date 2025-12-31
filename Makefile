.PHONY: nanochat nb blank

# Pull changes from karpathy's original nanochat repo.
nanochat:
	git fetch nanochat
	git subtree pull --prefix=nanochat nanochat master --squash

nb:
	cp -i notebooks/TEMPLATE.ipynb notebooks/todo_untitled.ipynb

# "commit notes" even though we technically don't commit these, want commit history to
# distinguish between days with notes updates and no updates
cnotes:
	# Changes to _notes don't show up in staging area because it's gitignored.
	git commit -m "notes" --allow-empty
