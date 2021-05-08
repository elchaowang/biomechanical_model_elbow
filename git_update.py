from git import Repo
import os


dirFile = os.path.abspath('')
repo = Repo(dirFile)
g = repo.git
g.pull()



