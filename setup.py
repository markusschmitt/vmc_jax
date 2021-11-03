import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='jVMC',
    version='0.1.1',
    author="Markus Schmitt, Moritz Reh",
    author_email="markus.schmitt@uni-koeln.de",
    description="jVMC: Versatile and performant variational Monte Carlo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://jvmc.readthedocs.io/en/latest/#",
    packages=setuptools.find_packages(),
    install_requires=["setuptools", "wheel", "numpy", "jax==0.2.24", "jaxlib==0.1.73", "flax==0.3.6", "mpi4py", "h5py"],
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
)
