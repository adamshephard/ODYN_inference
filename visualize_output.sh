#! /bin/bash

# Visualize output with TIAViz.
# Give directories of slides and overlays below.
# Loads TIAViz locally in a browser window given by port 5006.

slide_dir="/data/ANTICIPATE/github/testdata/wsis/"
overlay_dir="/data/ANTICIPATE/github/testdata/output/odyn/"

tiatoolbox visualize --slides $slide_dir --overlays $overlay_dir --port 5006