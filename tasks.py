from invoke import Collection, task



@task
def docs(c,builddir=None):
    env={}
    if builddir:
        env["BUILDDIR"]=builddir
    with c.cd('docs'):
        c.run(command='make html', env=env)

@task
def docs_view(c,port=9988):
    c.run(command=f'python -m http.server --directory docs/_build/html {port}', pty=True)

@task
def test(c):
    c.run('pytest tests/')

@task
def lint(c):
    c.run('validate-pyproject pyproject.toml')

@task
def wheel(c,wheelhouse="wheelhouse"):
    c.run(f'pip wheel -w {wheelhouse} --no-deps -v .')

