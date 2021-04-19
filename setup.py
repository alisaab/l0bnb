from setuptools import setup, find_packages

LICENSE = """MIT"""


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name="l0bnb",
    description="Scalable algorithms for L0 L2-regularized regression",
    author="Hussein Hazimeh, Rahul Mazumder, Ali Saab",
    author_email="alikassemsaab@gmail.com",
    url='https://github.com/alisaab/l0bnb',
    download_url="https://github.com/alisaab/l0bnb/archive/0.1.0.tar.gz",
    install_requires=["numpy >= 1.18.1", "scipy >= 1.4.1", "numba >= 0.53.1"],
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    license=LICENSE,
    long_description=readme(),
    long_description_content_type="text/markdown"
)
