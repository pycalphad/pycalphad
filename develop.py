import toml
import subprocess
import sys
import pathlib

path_to_develop = pathlib.Path(__file__).parent.absolute()

c = toml.load(path_to_develop / "pyproject.toml")
build_deps = c["build-system"]["requires"]
if __name__ == '__main__':
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet"] + build_deps)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--editable", str(path_to_develop)])
