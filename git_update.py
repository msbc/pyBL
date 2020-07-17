import os


def git_pull():
    try:
        import git
        git.Repo(os.path.expanduser(os.path.split(__file__)[0])).remote().pull()
    except:
        print('git pull FAILED!')
