from setuptools import find_packages, setup

setup(
    name="htm",
    version="0.1.0",
    description="Herbarium Time Machine",
    author="Esteban Esquivel",
    author_email="eadan97@gmail.com",
    url="https://github.com/eadan97/herbarium-time-machine",
    install_requires=["pytorch-lightning>=1.2.0", "hydra-core>=1.0.6"],
    packages=find_packages(
        where='src',
    ),
    package_dir={"": "src"}
    # packages=['htm'],
    # package_dir={'htm':'src'}
)