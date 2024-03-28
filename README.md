# 3D Printed Map project


This repo contains code and resources I used to create a _printable_ 3D map model.


(This document will be further detailed)


In a nutshell : 


 * I retrieve DEM data (in this case from the IGN)
 * I load any kind of geometry data (here hiking trail)
 * Rasterise the latter onto the DEM raster
 * Use PIL to add text to the DEM raster
 * Create a 3D model based on the modified DEM data


What's next?


 * The quality of the extruded text is poor...
 * Work on generating a 3D mesh of the text and then add it to the map 3D model