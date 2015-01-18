# laserlib
Library for 3D LiDAR perception

Author: Alastair Quadros

LaserLib is a c++ library containing work from my PhD thesis on processing 3D point clouds from a Velodyne LiDAR for object classification.

It is developed for close use with python and numpy: all functions have a python wrapper.

It consists of:
- Datastructures for selecting regions of points on range images / point clouds
- Surface normal routines
- Features such as PCA, spin images, line images, interfaces to PCL features
- Knn classification
- Affinity propagation clustering

The best documentation is in the python interfaces, which largely mirror the c++ ones.
http://www-personal.acfr.usyd.edu.au/a.quadros/LaserPy/index.html

Dependencies
------------

- Eigen 3 http://eigen.tuxfamily.org/
- Boost

Optional, recommended:
- Python (>2.7)
- Flann http://www.cs.ubc.ca/research/flann/

Optional:
- PCL (Point Cloud Library) http://www.pointclouds.org/


Relevant Publications
---------------------

@CONFERENCE{quadros2012feature,  
    author = {Quadros, A. and Underwood, J.P. and Douillard, B.},  
    title = {An Occlusion-aware Feature for Range Images},  
    booktitle = {Robotics and Automation, 2012. ICRA'12. IEEE International Conference on},  
    year = {2012},  
    month = {May 14-18},  
    organization = {IEEE}  
}

http://db.acfr.usyd.edu.au/download.php/Quadros2012ICRA_OcclusionAware.pdf?id=2522

@PHDTHESIS{quadros2014  
    author = {Quadros, A},  
    title = {Representing 3D shape in sparse range images for urban object classification},  
    year = {2014},  
    school = {The University of Sydney}  
}

http://hdl.handle.net/2123/10515

