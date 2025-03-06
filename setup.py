#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="syn-ct-from-mri",
    version="1.0.0",
    author="VN-Radiotherapy",
    author_email="contact@example.com",
    description="Chuyển đổi MRI sang CT mô phỏng cho xạ trị",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/syn-ct-from-mri-radiotherapy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "syn-mri2ct=main:main",
        ],
    },
) 