import git

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
    __version__ += ".dev" + str(n_commits) + "+" + sha[:7]
if repo.is_dirty():
    __version__ += "+dirty"