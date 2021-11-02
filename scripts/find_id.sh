#!/bin/bash
find results/ -name all* | awk -F/ '{print $5}'
