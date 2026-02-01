"""
Setup script for Face Recognition eKYC
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="face-ekyc",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Face Recognition system for electronic Know Your Customer (eKYC)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/face-ekyc",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=7.0.0",
            "mypy>=1.0.0",
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "face-ekyc=api.main:main",
        ],
    },
)
