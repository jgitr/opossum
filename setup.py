from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_args = dict(
    name='opossum',
    version='0.1.6',
    description='Simulated Data Generating Process',
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(),
    license='MIT',
    url="https://humboldt-wi.github.io/blog/research/",
    author='Tobias Krebs, Julian Winkel',
    author_email='julian.winkel@hu-berlin.de'
)

install_requires = [
    'numpy',
    'statsmodels',
    'seaborn',
    'matplotlib',
    'scipy'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)


