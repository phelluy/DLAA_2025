# small script to clean ipynb notebooks
# so that they can be displayed in Github

import nbformat

# Load notebook
nb_path = "check_training.ipynb"
nb = nbformat.read(nb_path, as_version=4)

# Remove widget metadata
if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

# Save notebook
nbformat.write(nb, nb_path)
print("Widget metadata removed.")
