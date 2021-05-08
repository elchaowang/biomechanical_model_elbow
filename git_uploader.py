from git import Repo
import os


def get_all_scripts(PATH):
    files = os.listdir(dirFile)
    scripts = list()
    for item in files:
        if item[-2:] == 'py':
            scripts.append(item)
    print(scripts)
    return scripts


dirFile = os.path.abspath('')
# print(dirFile)
repo = Repo(dirFile)
g = repo.git

py_scripts = get_all_scripts(dirFile)
for script in py_scripts:
    g.add(script)

g.commit("-m script auto update")
g.push()
print('success~')



