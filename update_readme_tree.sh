#!/bin/bash

# Create a new tree block
echo '```text' > tree.md
{
  echo '```text'
  echo "├── README.md"
  find . -maxdepth 1 -type d ! -name "." | sort | sed 's|^\./|├── |'
  echo '```'
} > tree.md
echo '```' >> tree.md

# Replace the tree block in README.md between TREE START and TREE END markers
awk '
/<!-- TREE START -->/ {print; system("cat tree.md"); skip=1}

skip==0 {print}
' README.md > temp.md && mv temp.md README.md