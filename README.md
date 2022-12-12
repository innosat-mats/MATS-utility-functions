# MATS-utility-functions
A repository containing utility functions for processing and visualising MATS data

## Description

This repository contains several modules practical for use with MATS data. 

geolocation:
imagetools:
plotting:
rawdata:
retrieval:
selection_tools:

## Installation

### Installation requirements

MATS-L1-processing


### Installation

1. Make sure you have pip installed in your current envirnoment (e.g. $conda install pip )

2. Run $pip install . or $pip install -e . if you want to do development for the package

3. run pytest by typing "pytest" in root folder

### Detailed instruction for Windows

    Download Anaconda navigator and update to newest version

    Download git for windows

    Make user account on github.com

    Make ssh-key in git-bash:

        $ssh-keygen -t ed25519 -C "user@mail.com" copy keys to .ssh folder in user home directory add config file to .ssh in user home directory

            "Host github.com IdentityFile ~.ssh/github"

        test with $ssh -T git@github.com add public key to github user preferences

    Setup user in git-bash:

        $git config --global user.name "UserName" $git config --global user.email "email@mail.com"

    Clone repository

        $git clone git@github.com:innosat-mats/MATS-utility-functions.git

    Setup conda environment

        $conda create -n python=3.9 $conda install pip

    Install package

        $pip install -e .

    Test module

        $pytest
        
### Known issues

Cartopy installation will fail if you do not have all the system libraries required for cartopy installed. Either install cartopy binary direct using 
$conda install cartopy

or install the required libraries, e.g. $sudo apt -y install libgeos-dev


Note

This project has been set up using PyScaffold 4.0.1. For details and usage information on PyScaffold see https://pyscaffold.org/.
