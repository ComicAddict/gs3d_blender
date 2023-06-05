# A Modular Approach for Creation of Any Bi-Axial Woven Structure with Congruent Tiles

Within this repository Blender programs and Geometry Nodes feature is used to develop the congruent tiles. 

```Models``` directory contains 3D printable .stl files of the shapes with slots.

```boundary_gen.blend``` is used for generating the 

# ABGeometry Controls == 
SUBDIV_LEVEL_SIMPLE: 
    Controls the simple subdivision used for subdividing the bounday curve
    
SUBDIV_LEVEL_SMOOTH: 
    Controls the catmull-clark subdivision used for subdividing the boundary curve
    
BOUNDARY_A:
    Controls the A parameter(i.e. height difference) of the boundary curve
    
BOUNDARY_B:
    Controls the B parameter(i.e. width difference) of the boundary curve
    
CUBE_VERTEX_AMOUNT:
    Controls the amound of verticec in the fundamental cube region, this might be needed for finer displacement of the shape later on if needed.


== ABCStructure Controls ==
SLAB_THICKNESS:
    Controls the height of the fundamental tile
    
N:
    Number of warp and weft threads, single value creates a square fabric pattern
    
A:
    Number of warps before wefts
    
B: 
    Number of wefts after warps
    
C:
    Shift amount at each different row
    
HIDE_WARP:
    Hides warp threads
    
HIDE_WEFT:
    Hides weft threads
    
SHOW_SINGLE_WARP_THREAD:
    Shows only a single warp thread defined by the WARP_THREAD_NUMBER=[0,N-1]
    
WARP_THREAD_NUMBER:
    Warp thread to be shown if SHOW_SINGLE_WARP_THREAD is selected

SHOW_SINGLE_WEFT_THREAD:
    Shows only a single weft thread defined by the WEFT_THREAD_NUMBER=[0,N-1]
    
WEFT_THREAD_NUMBER:
    Weft thread to be shown if SHOW_SINGLE_WEFT_THREAD is selected


"""