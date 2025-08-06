"""
Setup script for SafeMath library.
"""

from setuptools import setup, find_packages

setup(
    name="safemath",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'safemath=safemath.cli:main',
        ],
    },
)
