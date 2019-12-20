import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="agb94", # Replace with your own username
    version="0.0.1",
    author="Gabin An",
    author_email="agb94@kaist.ac.kr",
    description="SBFL package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agb94/sbfl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
       'numpy',
       'sklearn',
       'pandas',
       'imblearn',
       'scipy',
       'keras'
    ]
)
