import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pluribus-fedden",  # Replace with your own username
    version="0.0.1",
    author="Leon Fedden",
    author_email="leonfedden@gmail.com",
    description="Open source implementation of the pluribus poker AI ployer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fedden/pluribus",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
