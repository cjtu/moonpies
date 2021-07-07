#!/bin/bash
rm moonpies_package.zip

mkdir moonpies_package
mkdir moonpies_package/moonpies
mkdir moonpies_package/data
mkdir moonpies_package/figs

cp README.pdf moonpies_package

cp moonpies/__init__.py moonpies_package/moonpies
cp moonpies/default_config.py moonpies_package/moonpies
cp moonpies/moonpies.py moonpies_package/moonpies
cp moonpies/my_config.py moonpies_package/moonpies
cp moonpies/ensemble_plot.py moonpies_package/moonpies

cp data/crater_list.csv moonpies_package/data
cp data/needham_kring_2017.csv moonpies_package/data
cp data/costello_etal_2018_t1.csv moonpies_package/data

zip -r moonpies_package.zip moonpies_package

rm -r moonpies_package