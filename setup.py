from setuptools import find_packages, setup

with open("README.md") as f:
    README = f.read()

version = {}
# manually read version from file
with open("multiml/version.py") as file:
    exec(file.read())

setup(
    # some basic project information
    name="multiml",
    version=version,
    license="GPL3",
    description="Example python project",
    long_description=README,
    author="Evan Widloski",
    author_email="evan_multiml@widloski.com",
    url="https://github.com/evidlo/multiml",
    # your project's pip dependencies
    install_requires=[
        "numpy",
        "scikit-image",
        "tqdm"
    ],
    include_package_data=True,
    # automatically look for subfolders with __init__.py
    packages=find_packages(),
)
