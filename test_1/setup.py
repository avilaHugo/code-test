import platform
import sys

from setuptools import find_packages, setup

# Ensure the script only runs on Linux
if platform.system() != "Linux":
    sys.exit("This package can only be installed on Linux systems.")

# Read the requirements from rec.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="movieGetter",
    version="1.0.0",
    author="Connor",
    description="A CLI tool to fetch IMDb movie ratings and other info.",
    packages=find_packages(),
    install_requires=requirements,
    scripts=["MovieDetailgetter.py"],
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.12",
)
