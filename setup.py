import os

from setuptools import setup

PACKAGE_ROOT = 'kap'


def get_version(version_file='_version.py'):
    import importlib.util
    version_file_path = os.path.abspath(os.path.join(PACKAGE_ROOT, version_file))
    spec = importlib.util.spec_from_file_location('_version', version_file_path)
    version_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version_module)
    return str(version_module.__version__)


def setup_package():
    setup(version=get_version())


if __name__ == '__main__':
    setup_package()
