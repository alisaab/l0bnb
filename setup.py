from setuptools import setup, find_packages

LICENSE = """MIT"""


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name="l0bnb",
    description="least squares regression with l0l2 regularization",
    author="Ali Saab",
    author_email="alikassemsaab@gmail.com",
    install_requires=["numpy", "scipy", "numba", "matplotlib"],
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    license=LICENSE,
    long_description=readme(),
)
