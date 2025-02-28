import pathlib

from setuptools import setup


CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the mo-gymnasium version."""
    path = CWD / "morl_baselines" / "__init__.py"
    content = path.read_text()
    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


setup(
    name="morl-baselines",
    version=get_version(),
    description="Implementations of multi-objective reinforcement learning (MORL) algorithms.",
    long_description=open("README.md").read(),
)
