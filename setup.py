from setuptools import setup, find_packages

setup(
    name="easyvizar-identify",
    version="0.1",
    description="Person identification for EasyVizAR headsets",
    url="https://github.com/EasyVizAR/identify/",

    project_urls = {
        "Homepage": "https://wings.cs.wisc.edu/easyvizar/",
        "Source": "https://github.com/EasyVizAR/identify/",
    },

    packages=find_packages(),

    entry_points={
        "console_scripts": [
            "identify = identify.__main__:main"
        ]
    }
)
