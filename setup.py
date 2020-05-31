import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="poker_ai",
    version="1.0.0",
    author="Leon Fedden, Colin Manko",
    author_email="leonfedden@gmail.com",
    description="Open source implementation of a CFR based poker AI player.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fedden/poker_ai",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
