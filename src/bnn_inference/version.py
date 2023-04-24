import git

"""
Get version from git tags and commits.

If there are no tags, use 0.0.0.
Use the latest tag, and append the number of commits since that tag and the hash.
Indicate if the repo is dirty.

Example:
    0.0.0.dev3+af59d6d.dirty  - 3 commits since tag 0.0.0, hash af59d6d, repo is dirty
    0.0.0+af59d6d.dirty       - no tags, hash af59d6d, repo is dirty
    0.0.0                     - no tags, no commits, repo is clean
    1.0.0                     - tag 1.0.0, no commits, repo is clean
"""

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)

if len(tags) > 0:
    latest_tag = tags[-1]
    # get number of commits since latest tag
    commits_since_tag = repo.iter_commits(latest_tag.commit.hexsha + "..HEAD")
    n_commits = sum(1 for c in commits_since_tag)

    # Build version string with the tag + dev + number of commits since tag and hash
    __version__ = latest_tag.name
else:
    # No tags, use 0.0.0
    __version__ = "0.0.0"
    # Get number of commits since first commit
    commits_since_tag = repo.iter_commits()
    n_commits = sum(1 for c in commits_since_tag)
if n_commits > 0:
    __version__ += "+" + sha[:7] + "dev" + str(n_commits)
if repo.is_dirty():
    __version__ += "dirty"
