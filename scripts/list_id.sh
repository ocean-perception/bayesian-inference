#!/bin/bash
find results/ -name net* | awk -F/ '{print $5}'
