from setuptools import setup, find_packages

LICENSE = """MIT"""


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name="l0bnb",
    description="least squares regression with l0l2 regularization",
    author="Ali Saab, Hussein Hazimeh, Rahul Mazumder",
    author_email="alikassemsaab@gmail.com",
    url='https://github.com/alisaab/l0bnb',
    download_url="https://github.com/alisaab/l0bnb/archive/0.0.1.tar.gz",
    install_requires=["numpy >= 1.18.2", "scipy >= 1.4.1", "numba >= 0.48.0",
                      "matplotlib >= 3.2.1"],
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    license=LICENSE,
    long_description=readme(),
)
