All contributions and ideas are very welcome.

If you want to work on the codebase, then although you don't have to, it is recommended to follow these prodedures to make collaborating with the codebase as painless as possible. 

### 1. Clone the repo

With ssh (recommended):
```bash
git clone git@github.com:fedden/poker_ai.git
```

With https:
```bash
git clone https://github.com/fedden/poker_ai.git
```

### 2. Initialise the repo with git flow

The author use the awesome [git flow](https://github.com/nvie/gitflow), which can make [collaborating with other users a lot easier](https://nvie.com/posts/a-successful-git-branching-model/). Essentially is a collection of high level git scripts that encourage branching.

Should you also want to make use of this, then follow the instructions to get it set up with the repo. You don't have to use it, but this guide will assuse you'll adopt the tools, other wise find and use the relevant git commands to do the same things.

Installing on Debian/Ubuntu
```bash
sudo apt install git-flow
```

If you are using Mac OS X, please checkout the [install instructions](https://github.com/nvie/gitflow/wiki/Mac-OS-X).

Now `cd` to the root of the poker_ai repo.

Run:
```bash
git flow init
```

This will now prompt you to set up your branches. Ensure that the "branch should be used for bringing forth production releases" is `master`, and that the "branch should be used for integration of the next release" is `develop`. Hit enter for the rest of the options to provide the defaults. Your terminal should look something like this:

```bash
$ git flow init

Which branch should be used for bringing forth production releases?
   - develop
   - master
Branch name for production releases: [master] master

Which branch should be used for integration of the "next release"?
   - develop
Branch name for "next release" development: [develop] develop

How to name your supporting branch prefixes?
Feature branches? [feature/]
Release branches? [release/]
Hotfix branches? [hotfix/]
Support branches? [support/]
Version tag prefix? []
```

Once you have done this, you are ready to create feature branches to add your new feature.

Feature branches are always created based of of the `develop` branch, which is our "production" branch. Periodically we will merge `develop` into `master` and this will be our stable release, for users only. To create a feature branch, run: 
```
git flow feature start my-feature-branch-name-here
```

You are now in your branch, add and commit to it like normal.
```bash
git add .
git commit -m 'commit message here'
git push origin feature/my-feature-branch-name-here
```

### 3. Working on your feature branch

You should be based off of branch `develop` whether or not you make use of `git flow`. Commit your code like normal, and if there has been a day or more between your last commit, you may need to rebase your changes on top of the latest commits (head) of `develop`. 

First fetch the latest changes from all branches (and prune any deleted branches):
```bash
git fetch origin -p
```

Next ensure your local `develop` has all of the changes that the remote `develop` has.
```bash
git rebase origin/develop develop
```

Finally ensure your feature branch has all of the changes in `develop` in it.
```bash
git rebase develop feature/my-feature-branch-name-here
```

When you rebase `develop` into your feature branch, you will need to force-push it to the repo. PLEASE BE EXTRA CAREFUL with this - only use force push on a feature branch that only you have worked on, otherwise you may overwrite other peoples commits, as it will directly modify the repo's git history. 
```bash
git push origin feature/my-feature-branch-name-here --force
```

### 4. Creating a PR

Once you have wrote your cool new feature, you'll need to make a PR. If you can write any tests to support any new features introduced, this would be very welcome. If you have any conflicts with `develop` please ensure you have rebased with `develop` as instructed in step (3).

Please open a pull request via the github UI and request to merge into `develop`. Once there has been a successful review of your PR, and the automated tests pass, then feel free to merge at your leisure.
