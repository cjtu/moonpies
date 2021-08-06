#!/bin/bash

date=$(date '+%y%m%d')
pkg=dist/moonpies_package_${date}.zip
rm $pkg

mkdir moonpies_package
mkdir moonpies_package/moonpies
mkdir moonpies_package/data
mkdir moonpies_package/figures
mkdir moonpies_package/test

cp README.pdf moonpies_package

# Code files
cp moonpies/__init__.py moonpies_package/moonpies
cp moonpies/default_config.py moonpies_package/moonpies
cp moonpies/moonpies.py moonpies_package/moonpies
cp moonpies/my_config.py moonpies_package/moonpies
cp moonpies/ensemble_plot.py moonpies_package/moonpies

# Data files
cp data/crater_list.csv moonpies_package/data
cp data/basin_list.csv moonpies_package/data
cp data/needham_kring_2017_s3.csv moonpies_package/data
cp data/costello_etal_2018_t1.csv moonpies_package/data
cp data/bahcall_etal_2001_t2.csv moonpies_package/data
cp data/ballistic_sed_teq.csv moonpies_package/data
cp data/ballistic_hop_coldtraps.csv moonpies_package/data

# Test files
cp test/test_moonpies.py moonpies_package/test

zip -r $pkg moonpies_package

rm -r moonpies_package
