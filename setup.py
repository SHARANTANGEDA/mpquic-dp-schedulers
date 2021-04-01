from setuptools import setup, find_packages

REQUIRED_PACKAGES = list(filter(None, list(map(lambda s: s.strip(), open('requirements.txt').readlines()))))

with open("README.md", "r") as readme:
    long_description = readme.read()
setup(
    name='mpquic_schedulers',
    version='1.0',
    author="Sai Sharan Tangeda",
    author_email="saisarantangeda@gmail.com",
    description="Repo containing MP-QUIC Schedulers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SHARANTANGEDA/mpquic-dp-schedulers",
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 2.7'
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
