# GIT CHEATSHEET 1.0

Essential list of git commands to work safely with code versioning tools.

[Official Reference](https://git-scm.com/docs)

+ Clone repository locally: `git clone <https_path.git>`
+ Check local changes: `git status`
+ See differences between remote and local file: `git diff <filepath>`
+ List differing files between current and other branche: `git diff --name-only <other_branch>`
+ List differences in specific file between current and other branch: `git diff <other_branch> <filename>`
+ Stage local file edits: `git add <filepath>`
+ Undo local file edits before adding: `git reset <filepath>`
+ Commit added file: `git commit -m "Specific description"`
+ Align local to remote: `git pull origin <remote_branch>`
+ Align remote to local: `git push origin <remote_branch>`
+ Create new local branch and switch to it: `git checkout -b <new_branch_name>`
+ Creare new local branch and align to existing remote branch: `git chieckout -b <new_local_branc> origin/<remote_branch>`
+ Switch branch locally: `git checkout <destination_branch_name>`
+ Cherrypick files from another local branch: `git checkout <other branch> <space_separated_files>`
+ List all local branches: `git branch`

