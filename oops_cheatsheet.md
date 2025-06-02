# Oops Recovery Cheatsheet for Aircraft Classifier

This document explains how to fix common Git mistakes. Use it to recover from accidental file commits, bad pushes, or overwritten files.

### 1. Accidentally Committed a File You Shouldn't Have

To stop tracking a file but keep it on your computer, run:  
git rm --cached path/to/file.txt

Then add the file to `.gitignore`:  
echo "path/to/file.txt" >> .gitignore

Finally, commit and push:  
git add .gitignore  
git commit -m "Remove accidental file and update .gitignore"  
git push

### 2. Overwrote a Good File (e.g., README.md)

To recover a previous version of a file:

Run git log path/to/file to find a good commit hash  
Restore it using: git checkout <commit_hash> -- path/to/file  
Then commit and push:  
git commit -m "Restore correct version of file"  
git push

### 3. Undo the Last Commit (Before Pushing)

If you want to undo the last commit but keep your changes:  
Use git reset --soft HEAD~1 to keep changes staged  
Use git reset --mixed HEAD~1 to keep changes unstaged

### 4. Pushed Something You Shouldn’t Have

To undo a bad commit after pushing:

Use git revert <bad_commit_hash> to undo with a new commit  
Or use git reset --hard <good_commit_hash> followed by git push --force  
Only use force push if coordinated with your team

### 5. Not Sure What You Did? Use Git Reflog

Run git reflog to see your recent Git actions  
Reset to a previous state with: git reset --hard HEAD@{n}

### 6. Committed a Secret or Password

If you accidentally committed sensitive information:

Stop pushing further changes  
Contact the project lead immediately  
Use git filter-repo or BFG Repo Cleaner to remove secrets from history  
Rotate any affected credentials

### Best Practices

Make small, frequent commits  
Add logs, keys, and environment files to `.gitignore` early  
Avoid force pushes unless absolutely necessary  
Use pull requests and code reviews when working with shared branches like `main`
