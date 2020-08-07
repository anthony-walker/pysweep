import setuptools

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="pysweep",
    version="0.0.1",
    author="Anthony Walker",
    author_email="walkanth@oregonstate.edu",
    license='BSD 3-clause "New" or "Revised License"',
    description="This package is used for solving PDEs on distributed computing systems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anthony-walker/pysweep-git",
    entry_points={
        'console_scripts': [
            'pysweep=pysweep.utils.commandline:commandLine',
            ]
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    # package_data={'pyplume': ['mechanisms/*','originals/*','tests/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['pytest','h5py','matplotlib']
)
