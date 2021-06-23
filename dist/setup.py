import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='jVMC',
    version='1.0',
    scripts=['jvmc_exec'],
    author="Markus Schmitt",
    author_email="markus.schmitt@uni-koeln.de",
    description="jVMC: Versatile and performant variational Monte Carlo",
    long_description="This package, available on GitHub, provides a versatile and efficient implementation of variational quantum Monte Carlo in Python. It utilizes Google’s JAX library to exploit the blessings of automatic differentiation and just-in-time compilation for available computing resources. The package is devised to serve as white box implementation of the core computational tasks that guarantees efficiency, while at the same time providing large flexibility.",
    long_description_content_type="text/markdown",
    url="https://jvmc.readthedocs.io/en/latest/#",
    packages=setuptools.find_packages(),
    install_requires=["setuptools", "wheel", "numpy", "jax", "jaxlib", "flax", "mpi4py", "h5py"],
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
)
