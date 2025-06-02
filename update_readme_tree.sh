#!/bin/bash

# Create a new tree block
echo '```text' > tree.md
tree -L 3 >> tree.md
echo '```' >> tree.md

# Replace the tree block in README.md between TREE START and TREE END markers
awk '
/<!-- TREE START -->/ {print; system("cat tree.md"); skip=1}
/<!-- TREE END -->/ {skip=0; print; next}
skip==0 {print}
' README.md > temp.md && mv temp.md README.md