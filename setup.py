from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='directed_information',
      version='0.1',
      description='Universal Estimation of Directed Information',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/elipugh/directed_information',
      author='Eli Pugh',
      author_email='epugh@stanford.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=["tqdm","numpy"],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent"],
      keywords=["universal","directed","information","estimation","mutual"],
      zip_safe=False)

