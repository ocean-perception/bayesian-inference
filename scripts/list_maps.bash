#!/bin/bash
LIST=$(ls map* | sed 's/map_//g' | sed 's/\.csv//g')
