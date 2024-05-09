import bpy
import numpy as np
from collections import namedtuple
import random
from dataclasses import dataclass
import mathutils
import math

@dataclass
class nodeStruct:
    config: int
    pos: list[float]
    e1: list[list[float]]
    e2: list[list[float]]
    e3: list[list[float]]
    a: bool
@dataclass
class nodeStruct2d:
    config: bool
    pos: list[float]
    e1: list[list[float]]
    e2: list[list[float]]
aNode = nodeStruct(1,[0.0,0.0,0.0], [[0.0,0.0,0.1],[0.0,0.1,0.0]], [[0.1,0.0,0.1],[0.1,0.1,0.0]], [[0.0,0.1,0.1],[0.0,0.1,0.1]], False)

def getDispNodeX(config):
    match config:
        case 0:
            return np.array([0.0,1.0,1.0])
        case 1:
            return np.array([0.0, -1.0, 1.0])
        case 2:
            return np.array([0.0, 1.0, -1.0])
        case 3:
            return np.array([0.0, -1.0, -1.0])
        case _:
            return np.array([-1.0, -1.0, 0.0])
        
def getDispNodeX2d(config):
    match config:
        case 0:
            return np.array([0.0, 0.0, 1.])
        case 1:
            return np.array([0.0, 0.0, -1.])
        case _:
            return np.array([-1.0, -1.0, 0.0])
        
def getDispNodeY2d(config):
    match config:
        case 0:
            return np.array([0.0, 0.0, 1.])
        case 1:
            return np.array([0.0, 0.0, -1.])
        case _:
            return np.array([-1.0, -1.0, 0.0])

def getDispNodeY(config):
    match config:
        case 0:
            return np.array([1.0, 0.0, 1.0])
        case 1:
            return np.array([-1.0, 0.0, 1.0])
        case 2:
            return np.array([1.0, 0.0, -1.0])
        case 3:
            return np.array([-1.0, 0.0, -1.0])
        case _:
            return np.array([-1.0, -1.0, 0.0])

def getDispNodeZ(config): 
    match config:
        case 0:
            return np.array([1.0, 1.0, 0.0])
        case 1:
            return np.array([-1.0, 1.0, 0.0])
        case 2:
            return np.array([1.0, -1.0, 0.0])
        case 3:
            return np.array([-1.0, -1.0, 0.0])
        case _:
            return np.array([-1.0, -1.0, 0.0])

def randomizeStructure(nodes):
    for i in range(nodes.shape[0]):
        for j in range(nodes.shape[1]):
            for k in range(nodes.shape[2]):
                nodes[i,j,k].config = random.randint(0,8)


def abcStructure(nodes, config, a, b, c):
    s0 = 1
    s1 = 2
    for i in range(nodes.shape[0]):
        for j in range(nodes.shape[1]): 
            for k in range(nodes.shape[2]):
                if((i-c*(j + k)) % (a+b) < a):
                    nodes[i,j,k].config = 0
                    nodes[i,j,k].a = False
                else:
                    nodes[i,j,k].config = config
                    nodes[i,j,k].a = True

def abcStructure2d(nodes, config, a, b, c):
    for i in range(nodes.shape[0]):
        for j in range(nodes.shape[1]): 
            if((i-c*(j)) % (a+b) < a):
                nodes[i,j].config = 0
                nodes[i,j].a = False
            else:
                nodes[i,j].config = True
                nodes[i,j].a = True

def fabricShiftStructure(nodes, config, fabric, s0, s1):
    for i in range(nodes.shape[0]):
        for j in range(nodes.shape[1]): 
            for k in range(nodes.shape[2]):
                if(fabric[(i+k*s0)%fabric.shape[0],(j+k*s1)%fabric.shape[1]] == 0):
                    nodes[i,j,k].config = 0
                    nodes[i,j,k].a = False
                else:
                    nodes[i,j,k].config = config
                    nodes[i,j,k].a = True
                    
def abcShiftStructure(nodes, config, a, b, c, s0, s1):
    for i in range(nodes.shape[0]):
        for j in range(nodes.shape[1]): 
            for k in range(nodes.shape[2]):
                if((( i + (k* s0))-c*(j + (k * s1))) % (a+b) < a):
                    nodes[i,j,k].config = 0
                    nodes[i,j,k].a = False
                else:
                    nodes[i,j,k].config = config
                    nodes[i,j,k].a = True
                    """
    for k in range(nodes.shape[2]):
        for i in range(nodes.shape[0]): 
            for j in range(nodes.shape[1]):
                if k > 0:
                    nodes[i,j,k].config = nodes[(i - k * s0) % nodes.shape[0], (j - k * s1) % nodes.shape[1], k-1].config
                    print("Equating cell ", i,", ", j,", ", k , " to ", (i - k * s0) % nodes.shape[0], ", ", (j - k * s1) % nodes.shape[1], ", ", k - 1)
                    nodes[i,j,k].a = nodes[(i - k * s0) % nodes.shape[0], (j - k * s1) % nodes.shape[1], k-1].a
"""
def updateStructure(nodes, space):
    for i in range(nodes.shape[0]): 
        for j in range(nodes.shape[1]):
            for k in range(nodes.shape[2]):
                nodes[i,j,k].pos = np.array([i * space, j * space, k * space])
                nodes[i,j,k].e1[0] = space * getDispNodeX(0) / 4.0
                nodes[i,j,k].e1[1] = space * getDispNodeX(0) / 4.0
                nodes[i,j,k].e2[0] = space * getDispNodeY(2) / 4.0
                nodes[i,j,k].e2[1] = space * getDispNodeY(2) / 4.0
                nodes[i,j,k].e3[0] = space * getDispNodeZ(3) / 4.0
                nodes[i,j,k].e3[1] = space * getDispNodeZ(3) / 4.0

                if (nodes[i,j,k].config & (1 << 0)):
                    nodes[i,j,k].e1[0][2] *= -1.0
                    nodes[i,j,k].e1[1][2] *= -1.0
                    nodes[i,j,k].e2[0][2] *= -1.0
                    nodes[i,j,k].e2[1][2] *= -1.0
                    nodes[i,j,k].e3[0][2] *= -1.0
                    nodes[i,j,k].e3[1][2] *= -1.0
                

                if (nodes[i,j,k].config & (1 << 1)):
                    nodes[i,j,k].e1[0][1] *= -1.0
                    nodes[i,j,k].e1[1][1] *= -1.0
                    nodes[i,j,k].e2[0][1] *= -1.0
                    nodes[i,j,k].e2[1][1] *= -1.0
                    nodes[i,j,k].e3[0][1] *= -1.0
                    nodes[i,j,k].e3[1][1] *= -1.0

                if (nodes[i,j,k].config & (1 << 2)):
                    nodes[i,j,k].e1[0][0] *= -1.0
                    nodes[i,j,k].e1[1][0] *= -1.0
                    nodes[i,j,k].e2[0][0] *= -1.0
                    nodes[i,j,k].e2[1][0] *= -1.0
                    nodes[i,j,k].e3[0][0] *= -1.0
                    nodes[i,j,k].e3[1][0] *= -1.0
                
                if i == 0:
                    nodes[i,j,k].e1[0] += np.array([i * space - space / 2, j * space, k * space])
                else:
                    nodes[i,j,k].e1[0] += np.array([i * space - space / 4, j * space, k * space])
                if i == nodes.shape[0] - 1:
                    nodes[i,j,k].e1[1] += np.array([i * space + space / 2, j * space, k * space])
                else:
                    nodes[i,j,k].e1[1] += np.array([i * space + space / 4, j * space, k * space])
                    
                if j == 0:
                    nodes[i,j,k].e2[0] += np.array([i * space, j * space - space / 2, k * space])
                else:
                    nodes[i,j,k].e2[0] += np.array([i * space, j * space - space / 4, k * space])
                if j == nodes.shape[1] - 1:
                    nodes[i,j,k].e2[1] += np.array([i * space, j * space + space / 2, k * space])
                else:
                    nodes[i,j,k].e2[1] += np.array([i * space, j * space + space / 4, k * space])
                    
                if k == 0:
                    nodes[i,j,k].e3[0] += np.array([i * space, j * space, k * space - space / 2])
                else:
                    nodes[i,j,k].e3[0] += np.array([i * space, j * space, k * space - space / 4])
                if k == nodes.shape[2] - 1:
                    nodes[i,j,k].e3[1] += np.array([i * space, j * space, k * space + space / 2])
                else:
                    nodes[i,j,k].e3[1] += np.array([i * space, j * space, k * space + space / 4])




            
def updateStructure2d(nodes, space):
    for i in range(nodes.shape[0]): 
        for j in range(nodes.shape[1]):
        
            nodes[i,j].pos = np.array([i * space, j * space, 0])
            nodes[i,j].e1[0] = space * getDispNodeX2d(0) / 4.0
            nodes[i,j].e1[1] = space * getDispNodeX2d(0) / 4.0
            nodes[i,j].e2[0] = space * getDispNodeY2d(1) / 4.0
            nodes[i,j].e2[1] = space * getDispNodeY2d(1) / 4.0

            if (nodes[i,j].config):
                nodes[i,j].e1[0][2] *= -1.0
                nodes[i,j].e1[1][2] *= -1.0
                nodes[i,j].e2[0][2] *= -1.0
                nodes[i,j].e2[1][2] *= -1.0

   #         nodes[i,j].e1[0] += np.array([i * space - space / 4, j * space, 0])
  #          nodes[i,j].e1[1] += np.array([i * space + space / 4, j * space, 0])
 #           nodes[i,j].e2[0] += np.array([i * space, j * space - space / 4, 0])
#            nodes[i,j].e2[1] += np.array([i * space, j * space + space / 4, 0])
            
            if i == 0:
                nodes[i,j].e1[0] += np.array([i * space - space / 2, j * space,0])
            else:
                nodes[i,j].e1[0] += np.array([i * space - space / 4, j * space, 0])
            if i == nodes.shape[0] - 1:
                nodes[i,j].e1[1] += np.array([i * space + space / 2, j * space, 0])
            else:
                nodes[i,j].e1[1] += np.array([i * space + space / 4, j * space, 0])
                
            if j == 0:
                nodes[i,j].e2[0] += np.array([i * space, j * space - space / 2, 0])
            else:
                nodes[i,j].e2[0] += np.array([i * space, j * space - space / 4, 0])
            if j == nodes.shape[1] - 1:
                nodes[i,j].e2[1] += np.array([i * space, j * space + space / 2, 0])
            else:
                nodes[i,j].e2[1] += np.array([i * space, j * space + space / 4, 0])
            


def generateRandomStructure( dims, space):
    nodes = np.ndarray(shape=(dims[0],dims[1],dims[2]),dtype=nodeStruct)
    for i in range(nodes.shape[0]): 
        for j in range(nodes.shape[1]):
            for k in range(nodes.shape[2]):
                nodes[i,j,k] = nodeStruct(0, [0.0,0.0,0.0], [[0.0,0.0,0.1],[0.0,0.1,0.0]], [[0.1,0.0,0.1],[0.1,0.1,0.0]], [[0.0,0.1,0.1],[0.0,0.1,0.1]], False)
    print(nodes[0,0,0].e1)
    randomizeStructure(nodes)
    updateStructure(nodes, space)
    return nodes

def generateFabricShiftStructure(n, dims, space, config, fabric, s0, s1):
    if n == 3:
        nodes = np.ndarray(shape=(dims[0],dims[1],dims[2]),dtype=nodeStruct)
        for i in range(nodes.shape[0]): 
            for j in range(nodes.shape[1]):
                for k in range(nodes.shape[2]):
                    nodes[i,j,k] = nodeStruct(0, [0.0,0.0,0.0], [[0.0,0.0,0.1],[0.0,0.1,0.0]], [[0.1,0.0,0.1],[0.1,0.1,0.0]], [[0.0,0.1,0.1],[0.0,0.1,0.1]], False)

        fabricShiftStructure(nodes, config, fabric, s0, s1)
        updateStructure(nodes, space)
        ob = generateStructureData(nodes)
        ma = max(dims[0], dims[1], dims[2])
        m = 1.0/(max(dims[0], dims[1], dims[2])*4.0)
        ob.modifiers["GeometryNodes"]["Input_8"] = m
        ob.data.update()
        ob.location = ob.location - mathutils.Vector((dims[0]/2 * space - space/2, dims[1]/2 * space- space/2, dims[2]/2* space- space/2))
        
def generateABCShiftStructure(n, dims, space, config, a, b, c, s0, s1):
    if n == 3:
        nodes = np.ndarray(shape=(dims[0],dims[1],dims[2]),dtype=nodeStruct)
        for i in range(nodes.shape[0]): 
            for j in range(nodes.shape[1]):
                for k in range(nodes.shape[2]):
                    nodes[i,j,k] = nodeStruct(0, [0.0,0.0,0.0], [[0.0,0.0,0.1],[0.0,0.1,0.0]], [[0.1,0.0,0.1],[0.1,0.1,0.0]], [[0.0,0.1,0.1],[0.0,0.1,0.1]], False)

        abcShiftStructure(nodes, config, a, b, c, s0, s1)
        updateStructure(nodes, space)
        ob = generateStructureData(nodes)
        ma = max(dims[0], dims[1], dims[2])
        m = 1.0/(max(dims[0], dims[1], dims[2])*4.0)
        ob.modifiers["GeometryNodes"]["Input_8"] = m
        ob.data.update()
        ob.location = ob.location - mathutils.Vector((dims[0]/2 * space - space/2, dims[1]/2 * space- space/2, dims[2]/2* space- space/2))
    if n == 2:
        nodes = np.ndarray(shape=(dims[0],dims[1]),dtype=nodeStruct2d)
        for i in range(nodes.shape[0]): 
            for j in range(nodes.shape[1]):
                nodes[i,j] = nodeStruct2d(0, [0.0,0.0,0.0], [[0.0,0.0,0.1],[0.0,0.1,0.0]], [[0.1,0.0,0.1],[0.1,0.1,0.0]])

        abcStructure2d(nodes, config, a, b, c)
        updateStructure2d(nodes, space)
        generateStructureData2d(nodes)

def generateABCStructure(n, dims, space, config, a, b, c):
    if n == 3:
        nodes = np.ndarray(shape=(dims[0],dims[1],dims[2]),dtype=nodeStruct)
        for i in range(nodes.shape[0]): 
            for j in range(nodes.shape[1]):
                for k in range(nodes.shape[2]):
                    nodes[i,j,k] = nodeStruct(0, [0.0,0.0,0.0], [[0.0,0.0,0.1],[0.0,0.1,0.0]], [[0.1,0.0,0.1],[0.1,0.1,0.0]], [[0.0,0.1,0.1],[0.0,0.1,0.1]], False)

        abcStructure(nodes, config, a, b, c)
        updateStructure(nodes, space)
        ob = generateStructureData(nodes)
        ma = max(dims[0], dims[1], dims[2])
        m = 1.0/(max(dims[0], dims[1], dims[2])*5.0)
        ob.modifiers["GeometryNodes"]["Input_8"] = m
        ob.data.update()
        ob.location = ob.location - mathutils.Vector((dims[0]/2 * space - space/2, dims[1]/2 * space- space/2, dims[2]/2* space- space/2))
        makeActive(ob)
        bpy.ops.object.material_slot_add()
        mat = bpy.data.materials.get('pipeshader')
        ob.material_slots[0].material = mat
        
    if n == 2:
        nodes = np.ndarray(shape=(dims[0],dims[1]),dtype=nodeStruct2d)
        for i in range(nodes.shape[0]): 
            for j in range(nodes.shape[1]):
                nodes[i,j] = nodeStruct2d(0, [0.0,0.0,0.0], [[0.0,0.0,0.1],[0.0,0.1,0.0]], [[0.1,0.0,0.1],[0.1,0.1,0.0]])

        abcStructure2d(nodes, config, a, b, c)
        updateStructure2d(nodes, space)
        ob = generateStructureData2d(nodes)
        ma = max(dims[0], dims[1], dims[2])
        m = 1.0/(max(dims[0], dims[1], dims[2])*2.1)
        #ob.modifiers["GeometryNodes"]["Input_6"] = False
        ob.modifiers["GeometryNodes"]["Input_8"] = m
        ob.modifiers["GeometryNodes"]["Input_6"][0] = m
        ob.modifiers["GeometryNodes"]["Input_7"][0] = -m
        for i in range(9):
            print(ob.modifiers["GeometryNodes"]["Input_"+str(i+2)])
            ob.modifiers["GeometryNodes"]["Input_2"] = False
        ob.data.update()
        ob.location = ob.location - mathutils.Vector((dims[0]/2 * space - space/2, dims[1]/2 * space- space/2, dims[2]/2* space- space/2))
        makeActive(ob)
        bpy.ops.object.material_slot_add()
        mat = bpy.data.materials.get('pipeshader')
        ob.material_slots[0].material = mat
        
def generateFundamentalStructure(fx, fy, fz, sx, sy, sz, space, shape, name="def"):
    nodes = np.ndarray(shape=(shape[0],shape[1],shape[2]),dtype=nodeStruct)
    configs = np.ndarray(shape=(2,2,2), dtype=np.int32)
    configs[0,0,0] = 4 * fx         + 2 * fy        + fz
    configs[1,0,0] = 4 * sx         + 2 * (not fy)  + (not fz)
    configs[0,1,0] = 4 * (not fx)   + 2 * sy        + (not fz)
    configs[1,1,0] = 4 * (not sx)   + 2 * (not sy)  + fz
    configs[0,0,1] = 4 * (not fx)   + 2 * (not fy)  + sz
    configs[1,0,1] = 4 * (not sx)   + 2 * (fy)      + (not sz)
    configs[0,1,1] = 4 * fx         + 2 * (not sy)  + (not sz)
    configs[1,1,1] = 4 * sx         + 2 * sy        + sz
    print(configs)
    for i in range(nodes.shape[0]): 
        for j in range(nodes.shape[1]):
            for k in range(nodes.shape[2]):
                nodes[i,j,k] = nodeStruct(configs[i%2,j%2,k%2], [0.0,0.0,0.0], [[0.0,0.0,0.1],[0.0,0.1,0.0]], [[0.1,0.0,0.1],[0.1,0.1,0.0]], [[0.0,0.1,0.1],[0.0,0.1,0.1]], False)

    updateStructure(nodes, space)
    ob = generateStructureData(nodes, ob_name=name)
    ma = max(shape[0], shape[1], shape[2])
    m = 1.0/(max(shape[0], shape[1], shape[2])*6.0)
    ob.modifiers["GeometryNodes"]["Input_8"] = m
    ob.modifiers["GeometryNodes"]["Input_9"] = True
    ob.modifiers["GeometryNodes"]["Input_12"] = 64
    ob.data.update()
    #ob.location = ob.location - mathutils.Vector((shape[0]/2 * space - space/2, shape[1]/2 * space- space/2, shape[2]/2* space- space/2))
    return ob
    
def generateFundamentalStructureExplicit(configs, space, shape, name="def"):
    nodes = np.ndarray(shape=(shape[0],shape[1],shape[2]),dtype=nodeStruct)
    print(configs)
    for i in range(nodes.shape[0]): 
        for j in range(nodes.shape[1]):
            for k in range(nodes.shape[2]):
                nodes[i,j,k] = nodeStruct(configs[i%2,j%2,k%2], [0.0,0.0,0.0], [[0.0,0.0,0.1],[0.0,0.1,0.0]], [[0.1,0.0,0.1],[0.1,0.1,0.0]], [[0.0,0.1,0.1],[0.0,0.1,0.1]], False)

    updateStructure(nodes, space)
    ob = generateStructureData(nodes, ob_name=name)
    ma = max(shape[0], shape[1], shape[2])
    m = 1.0/(max(shape[0], shape[1], shape[2])*6.0)
    ob.modifiers["GeometryNodes"]["Input_8"] = m
    ob.modifiers["GeometryNodes"]["Input_9"] = True
    ob.modifiers["GeometryNodes"]["Input_12"] = 64
    ob.data.update()
    ob.location = ob.location - mathutils.Vector((shape[0]/2 * space - space/2, shape[1]/2 * space- space/2, shape[2]/2* space- space/2))
    return ob

def generateABCStructure2d(dims, space, config, a, b, c):
    nodes = np.ndarray(shape=(dims[0],dims[1]),dtype=nodeStruct2d)
    for i in range(nodes.shape[0]): 
        for j in range(nodes.shape[1]):
            nodes[i,j] = nodeStruct2d(0, [0.0,0.0,0.0], [[0.0,0.0,0.1],[0.0,0.1,0.0]], [[0.1,0.0,0.1],[0.1,0.1,0.0]])

    abcStructure2d(nodes, config, a, b, c)
    updateStructure2d(nodes, space)
    return nodes

def mergeByThreshold(o, tr):
    makeActive(o)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold = tr)
    bpy.ops.object.mode_set(mode='OBJECT')

def addGeoNodes(o, name):
    makeActive(o)
    bpy.ops.object.modifier_add(type='NODES')
    bpy.context.object.modifiers["GeometryNodes"].node_group = bpy.data.node_groups.get(name)
    
def addBevel(o, seg, width):
    makeActive(o)
    bpy.ops.object.modifier_add(type='BEVEL')
    bpy.context.object.modifiers["Bevel"]
    print(dir(bpy.context.object.modifiers["Bevel"]))
    bpy.context.object.modifiers["Bevel"].segments = seg
    bpy.context.object.modifiers["Bevel"].width = width
    bpy.context.object.modifiers["Bevel"].angle_limit = 3.14/3
    
def makeActive(o):
    bpy.context.view_layer.objects.active = o

def joinAll():
    for ob in bpy.context.scene.objects:
        print(ob.name)
        if ob.type == 'MESH':
            ob.select_set(True)
            bpy.context.view_layer.objects.active = ob
        else:
            ob.select_set(False)
    bpy.ops.object.join()
    bpy.ops.object.convert(target='MESH') 
    
def generateStructureData(nodes, v = [], c = [], ob_name="def_name") :
    edgesx = []
    edgesy = []
    edgesz = []
    ex = []
    ey = []
    ez = []
    ix = 0
    iy = 0
    iz = 0
    nx = []
    ny = []
    nz = []
    n1x = []
    n1y = []
    n1z = []
    
    nxx = []
    nxy = []
    nxz = []
    
    nyx = []
    nyy = []
    nyz = []
    
    nzx = []
    nzy = []
    nzz = []
    
    itx = []
    ity = []
    itz = []
    cx = []
    cy = []
    cz = []
    ax = []
    ay = []
    az = []
    qx = []
    qy = []
    qz = []
    q = 0
    wx = []
    wy = []
    wz = []
    for i in range(nodes.shape[0]):
        for j in range(nodes.shape[1]):
            for k in range(nodes.shape[2]):
                if (i != 0):
                    edgesx.append(nodes[i - 1,j,k].e1[1].tolist())
                    edgesx.append(nodes[i,j,k].e1[0].tolist())
                    ex.append([ix, ix+1])
                    nx.append(k*nodes.shape[0]+j)
                    nx.append(k*nodes.shape[0]+j)
                    n1x.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i - 1)
                    n1x.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i - 1)
                    nxx.append(i * nodes.shape[1] * nodes.shape[2] + j * nodes.shape[2] + k)
                    nxx.append(i * nodes.shape[1] * nodes.shape[2] + j * nodes.shape[2] + k)
                    nyx.append(j * nodes.shape[0] * nodes.shape[2] + i * nodes.shape[2] + k)
                    nyx.append(j * nodes.shape[0] * nodes.shape[2] + i * nodes.shape[2] + k)
                    nzx.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    nzx.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    itx.append(j)
                    itx.append(j)
                    for p in nodes[i,j,k].pos: cx.append(float(p))
                    for p in nodes[i,j,k].pos: cx.append(float(p))
                    ax.append(nodes[i-1,j,k].a or nodes[i,j,k].a)
                    ax.append(nodes[i-1,j,k].a or nodes[i,j,k].a)
                    ix += 2
                    qx.append(q)
                    qx.append(q)
                    q += 1
                    wx.append(True)
                    wx.append(True)
                if (j != 0):
                    edgesy.append(nodes[i,j - 1,k].e2[1].tolist())
                    edgesy.append(nodes[i,j,k].e2[0].tolist())
                    ey.append([iy, iy+1])
                    ny.append(k*nodes.shape[0]+i)
                    ny.append(k*nodes.shape[0]+i)
                    n1y.append(k * nodes.shape[0] * nodes.shape[1] + (j - 1) * nodes.shape[0] + i)
                    n1y.append(k * nodes.shape[0] * nodes.shape[1] + (j - 1) * nodes.shape[0] + i)
                    nxy.append(i * nodes.shape[1] * nodes.shape[2] + j * nodes.shape[2] + k)
                    nxy.append(i * nodes.shape[1] * nodes.shape[2] + j * nodes.shape[2] + k)
                    nyy.append(j * nodes.shape[0] * nodes.shape[2] + i * nodes.shape[2] + k)
                    nyy.append(j * nodes.shape[0] * nodes.shape[2] + i * nodes.shape[2] + k)
                    nzy.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    nzy.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    ity.append(i)
                    ity.append(i)
                    for p in nodes[i,j,k].pos: cy.append(float(p))
                    for p in nodes[i,j,k].pos: cy.append(float(p))
                    ay.append(nodes[i,j-1,k].a or nodes[i,j,k].a)
                    ay.append(nodes[i,j-1,k].a or nodes[i,j,k].a)
                    iy += 2
                    qy.append(q)
                    qy.append(q)
                    q += 1
                    wy.append(True)
                    wy.append(True)
                if (k != 0):
                    edgesz.append(nodes[i,j,k - 1].e3[1].tolist())
                    edgesz.append(nodes[i,j,k].e3[0].tolist())
                    ez.append([iz, iz+1])
                    nz.append(i*nodes.shape[2]+j)
                    nz.append(i*nodes.shape[2]+j)
                    n1z.append((k - 1) * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    n1z.append((k - 1) * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    nxz.append(i * nodes.shape[1] * nodes.shape[2] + j * nodes.shape[2] + k)
                    nxz.append(i * nodes.shape[1] * nodes.shape[2] + j * nodes.shape[2] + k)
                    nyz.append(j * nodes.shape[0] * nodes.shape[2] + i * nodes.shape[2] + k)
                    nyz.append(j * nodes.shape[0] * nodes.shape[2] + i * nodes.shape[2] + k)
                    nzz.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    nzz.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    itz.append(i * nodes.shape[1] + j)
                    itz.append(i * nodes.shape[1] + j)
                    for p in nodes[i,j,k].pos: cz.append(float(p))
                    for p in nodes[i,j,k].pos: cz.append(float(p))
                    az.append(nodes[i,j,k-1].a or nodes[i,j,k].a)
                    az.append(nodes[i,j,k-1].a or nodes[i,j,k].a)
                    iz += 2
                    qz.append(q)
                    qz.append(q)
                    q += 1
                    wz.append(True)
                    wz.append(True)
                n1x.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                n1x.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                
                n1y.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                n1y.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)

                n1z.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                n1z.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                
                nzx.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                nzx.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                nzy.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                nzy.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                nzz.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                nzz.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                
                nxx.append(i * nodes.shape[1] * nodes.shape[2] + j * nodes.shape[2] + k)
                nxx.append(i * nodes.shape[1] * nodes.shape[2] + j * nodes.shape[2] + k)
                nxy.append(i * nodes.shape[1] * nodes.shape[2] + j * nodes.shape[2] + k)
                nxy.append(i * nodes.shape[1] * nodes.shape[2] + j * nodes.shape[2] + k)
                nxz.append(i * nodes.shape[1] * nodes.shape[2] + j * nodes.shape[2] + k)
                nxz.append(i * nodes.shape[1] * nodes.shape[2] + j * nodes.shape[2] + k)
                
                nyx.append(j * nodes.shape[0] * nodes.shape[2] + i * nodes.shape[2] + k)
                nyx.append(j * nodes.shape[0] * nodes.shape[2] + i * nodes.shape[2] + k)
                nyy.append(j * nodes.shape[0] * nodes.shape[2] + i * nodes.shape[2] + k)
                nyy.append(j * nodes.shape[0] * nodes.shape[2] + i * nodes.shape[2] + k)
                nyz.append(j * nodes.shape[0] * nodes.shape[2] + i * nodes.shape[2] + k)
                nyz.append(j * nodes.shape[0] * nodes.shape[2] + i * nodes.shape[2] + k)

                nx.append(k*nodes.shape[1]+j)
                nx.append(k*nodes.shape[1]+j)
                ny.append(k*nodes.shape[0]+i)
                ny.append(k*nodes.shape[0]+i)
                nz.append(i*nodes.shape[2]+j)
                nz.append(i*nodes.shape[2]+j)
                
                itx.append(j)
                itx.append(j)
                ity.append(i)
                ity.append(i)    
                itz.append(i * nodes.shape[1] + j)
                itz.append(i * nodes.shape[1] + j)
                
                for p in nodes[i,j,k].pos: cx.append(float(p))
                for p in nodes[i,j,k].pos: cx.append(float(p))
                for p in nodes[i,j,k].pos: cy.append(float(p))
                for p in nodes[i,j,k].pos: cy.append(float(p))
                for p in nodes[i,j,k].pos: cz.append(float(p))
                for p in nodes[i,j,k].pos: cz.append(float(p))
                ax.append(nodes[i,j,k].a)
                ax.append(nodes[i,j,k].a)
                ay.append(nodes[i,j,k].a)
                ay.append(nodes[i,j,k].a)
                az.append(nodes[i,j,k].a)
                az.append(nodes[i,j,k].a)
                edgesx.append(nodes[i,j,k].e1[0].tolist())
                edgesx.append(nodes[i,j,k].e1[1].tolist())
                ex.append([ix, ix+1])
                ix += 2
                edgesy.append(nodes[i,j,k].e2[0].tolist())
                edgesy.append(nodes[i,j,k].e2[1].tolist())
                ey.append([iy, iy+1])
                iy += 2
                edgesz.append(nodes[i,j,k].e3[0].tolist())
                edgesz.append(nodes[i,j,k].e3[1].tolist())
                ez.append([iz, iz+1])
                iz += 2
                qx.append(q)
                qx.append(q)
                q += 1
                qy.append(q)
                qy.append(q)
                q += 1
                qz.append(q)
                qz.append(q)
                q += 1
            
                wx.append(False)    
                wx.append(False)    
                wy.append(False)    
                wy.append(False)    
                wz.append(False)    
                wz.append(False)    
    faces = []
    # Create Mesh Datablock
    mesh1 = bpy.data.meshes.new("edges x")
    mesh1.from_pydata(edgesx, ex, faces)
    mesh1.attributes.new(name="n", type="INT", domain="POINT")
    mesh1.attributes.new(name="n1", type="INT", domain="POINT")
    mesh1.attributes.new(name="nx", type="INT", domain="POINT")
    mesh1.attributes.new(name="ny", type="INT", domain="POINT")
    mesh1.attributes.new(name="nz", type="INT", domain="POINT")
    mesh1.attributes.new(name="c", type="FLOAT_VECTOR", domain="POINT")
    mesh1.attributes.new(name="a", type="BOOLEAN", domain="POINT")
    mesh1.attributes.new(name="w", type="BOOLEAN", domain="POINT")
    mesh1.attributes.new(name="q", type="INT", domain="POINT")
    mesh1.attributes.new(name="it", type="INT", domain="POINT")
    mesh1.attributes["n"].data.foreach_set("value",nx)
    mesh1.attributes["n1"].data.foreach_set("value",n1x)
    mesh1.attributes["nx"].data.foreach_set("value",nxx)
    mesh1.attributes["ny"].data.foreach_set("value",nyx)
    mesh1.attributes["nz"].data.foreach_set("value",nzx)
    mesh1.attributes["c"].data.foreach_set("vector",cx)
    mesh1.attributes["a"].data.foreach_set("value",ax)
    mesh1.attributes["w"].data.foreach_set("value",wx)
    mesh1.attributes["q"].data.foreach_set("value",qx)
    mesh1.attributes["it"].data.foreach_set("value",itx)
    
    mesh2 = bpy.data.meshes.new("edges y")
    mesh2.from_pydata(edgesy, ey, faces)
    mesh2.attributes.new(name="n", type="INT", domain="POINT")
    mesh2.attributes.new(name="n1", type="INT", domain="POINT")
    mesh2.attributes.new(name="nx", type="INT", domain="POINT")
    mesh2.attributes.new(name="ny", type="INT", domain="POINT")
    mesh2.attributes.new(name="nz", type="INT", domain="POINT")
    mesh2.attributes.new(name="c", type="FLOAT_VECTOR", domain="POINT")
    mesh2.attributes.new(name="a", type="BOOLEAN", domain="POINT")
    mesh2.attributes.new(name="w", type="BOOLEAN", domain="POINT")
    mesh2.attributes.new(name="q", type="INT", domain="POINT")
    mesh2.attributes.new(name="it", type="INT", domain="POINT")
    mesh2.attributes["n"].data.foreach_set("value",ny)
    mesh2.attributes["n1"].data.foreach_set("value",n1y)
    mesh2.attributes["nx"].data.foreach_set("value",nxy)
    mesh2.attributes["ny"].data.foreach_set("value",nyy)
    mesh2.attributes["nz"].data.foreach_set("value",nzy)
    mesh2.attributes["c"].data.foreach_set("vector",cy)
    mesh2.attributes["a"].data.foreach_set("value",ay)
    mesh2.attributes["w"].data.foreach_set("value",wy)
    mesh2.attributes["q"].data.foreach_set("value",qy)
    mesh2.attributes["it"].data.foreach_set("value",ity)
        
    mesh3 = bpy.data.meshes.new("edges z")
    mesh3.from_pydata(edgesz, ez, faces)
    mesh3.attributes.new(name="n", type="INT", domain="POINT")
    mesh3.attributes.new(name="n1", type="INT", domain="POINT")
    mesh3.attributes.new(name="nx", type="INT", domain="POINT")
    mesh3.attributes.new(name="ny", type="INT", domain="POINT")
    mesh3.attributes.new(name="nz", type="INT", domain="POINT")
    mesh3.attributes.new(name="c", type="FLOAT_VECTOR", domain="POINT")
    mesh3.attributes.new(name="a", type="BOOLEAN", domain="POINT")
    mesh3.attributes.new(name="w", type="BOOLEAN", domain="POINT")
    mesh3.attributes.new(name="q", type="INT", domain="POINT")
    mesh3.attributes.new(name="it", type="INT", domain="POINT")
    mesh3.attributes["n"].data.foreach_set("value",nz)
    mesh3.attributes["n1"].data.foreach_set("value",n1z)
    mesh3.attributes["nx"].data.foreach_set("value",nxz)
    mesh3.attributes["ny"].data.foreach_set("value",nyz)
    mesh3.attributes["nz"].data.foreach_set("value",nzz)
    mesh3.attributes["c"].data.foreach_set("vector",cz)
    mesh3.attributes["a"].data.foreach_set("value",az)
    mesh3.attributes["w"].data.foreach_set("value",wz)
    mesh3.attributes["q"].data.foreach_set("value",qz)
    mesh3.attributes["it"].data.foreach_set("value",itz)

    # Create Object and link to scene
    ob1 = bpy.data.objects.new("edges x object", mesh1)
    vs = ob1.data.vertices
    ob1.data.vertex_colors.new()
    
    #ob1.select = True
    ob2 = bpy.data.objects.new("edges y object", mesh2)
    ob2.data.vertex_colors.new()
    #ob2.select = True
    ob3 = bpy.data.objects.new("edges z object", mesh3)
    ob3.data.vertex_colors.new()
    
    me = bpy.data.meshes.new(ob_name + "Mesh")
    ob4 = bpy.data.objects.new(ob_name, me)

    # Make a mesh from a list of vertices/edges/faces
    #ob3.select = True
    #bpy.ops.object.join
    bpy.context.scene.collection.objects.link(ob1)
    bpy.context.scene.collection.objects.link(ob2)
    bpy.context.scene.collection.objects.link(ob3)
    bpy.context.scene.collection.objects.link(ob4)
    i = 0
    obs = [ob1, ob2, ob3]
    cols = [[.6, .6, 1, 1],[1, .6, .6, 1],[.6, 1, .6, 1]]
    for ob in obs:
        makeActive(ob)
        mergeByThreshold(ob, 0.01)
        #addBevel(ob, 3, 0.1)
        colattr = bpy.context.object.data.color_attributes.new(name = "vcol", type="FLOAT_COLOR", domain='POINT')
        herr = bpy.context.object.data.color_attributes.new(name = "herr", type="FLOAT_COLOR", domain='POINT')
        for v_index in range(len(bpy.context.object.data.vertices)):
            colattr.data[v_index].color = cols[i]
        for v_index in range(len(bpy.context.object.data.vertices)):
            if((v_index / ( nodes.shape[0] * 3)) % 2 == 0):
                herr.data[v_index].color = [0.0,0.0,0.0,1.0]
            else:
                herr.data[v_index].color = [1.0,1.0,1.0,1.0]
            a = math.floor((v_index/(nodes.shape[0] *2)))
            a = (a % 2 == 0)
            a = v_index/len(bpy.context.object.data.vertices)
            herr.data[v_index].color = [a,a,a,1.0]
        i+=1
        
    addGeoNodes(ob4, "pipes")
    #addGeoNodes(ob4, "scale")
    coll1 = bpy.ops.collection.create(name  = "EdgeCollection")
    bpy.context.scene.collection.children.link(bpy.data.collections["EdgeCollection"])
    makeActive(ob1)
    bpy.context.scene.collection.objects.unlink(bpy.context.object)
    bpy.data.collections["EdgeCollection"].objects.link(ob1)
    makeActive(ob2)
    bpy.context.scene.collection.objects.unlink(bpy.context.object)
    bpy.data.collections["EdgeCollection"].objects.link(ob2)
    makeActive(ob3)
    bpy.context.scene.collection.objects.unlink(bpy.context.object)
    bpy.data.collections["EdgeCollection"].objects.link(ob3)
    #bpy.ops.object.hide_collection(bpy.data.collections["EdgeCollection"])
    bpy.data.collections["EdgeCollection"].hide_viewport=True
    bpy.data.collections["EdgeCollection"].hide_render=True
    makeActive(ob1)
    return ob4
    #print(bpy.context.object.modifiers)

    #print(dir(bpy.context.object.modifiers["GeometryNodes"]))
    

def generateStructureData2d(nodes, v = [], c = []) :
    edgesx = []
    edgesy = []
    ex = []
    ey = []
    ix = 0
    iy = 0
    nx = []
    ny = []
    n1x = []
    n2x = []
    
    n1y = []
    n2y = []
    cx = []
    cy = []
    ax = []
    ay = []
    qx = []
    qy = []
    itx = []
    ity = []
    q = 0
    for i in range(nodes.shape[0]):
        for j in range(nodes.shape[1]):
            if (i != 0):
                edgesx.append(nodes[i - 1,j].e1[1].tolist())
                edgesx.append(nodes[i,j].e1[0].tolist())
                ex.append([ix, ix+1])
                nx.append(j)
                nx.append(j)
                n1x.append(j * nodes.shape[0] + i - 1)
                n1x.append(j * nodes.shape[0] + i - 1)
                n2x.append(i * nodes.shape[1] + j)
                n2x.append(i * nodes.shape[1] + j)
                itx.append(j)
                itx.append(j)
                for p in nodes[i,j].pos: cx.append(float(p))
                for p in nodes[i,j].pos: cx.append(float(p))
                ix += 2
                qx.append(q)
                qx.append(q)
                q += 1
            if (j != 0):
                edgesy.append(nodes[i,j - 1].e2[1].tolist())
                edgesy.append(nodes[i,j].e2[0].tolist())
                ey.append([iy, iy+1])
                ny.append(i)
                ny.append(i)
                n1y.append(j * nodes.shape[0] + i)
                n1y.append(j * nodes.shape[0] + i)
                n2y.append(i * nodes.shape[1] + j)
                n2y.append(i * nodes.shape[1] + j)
                ity.append(i)
                ity.append(i)
                for p in nodes[i,j].pos: cy.append(float(p))
                for p in nodes[i,j].pos: cy.append(float(p))
                iy += 2
                qy.append(q)
                qy.append(q)
                q += 1
            n1x.append(j * nodes.shape[0] + i)
            n1x.append(j * nodes.shape[0] + i)
            n2x.append(i * nodes.shape[1] + j)
            n2x.append(i * nodes.shape[1] + j)
            n1y.append(j * nodes.shape[0] + i)
            n1y.append(j * nodes.shape[0] + i)
            n2y.append(i * nodes.shape[1] + j)
            n2y.append(i * nodes.shape[1] + j)
            nx.append(j)
            nx.append(j)
            ny.append(i)
            ny.append(i)
            itx.append(j)
            itx.append(j)
            ity.append(i)
            ity.append(i)
            for p in nodes[i,j].pos: cx.append(float(p))
            for p in nodes[i,j].pos: cx.append(float(p))
            for p in nodes[i,j].pos: cy.append(float(p))
            for p in nodes[i,j].pos: cy.append(float(p))
            edgesx.append(nodes[i,j].e1[0].tolist())
            edgesx.append(nodes[i,j].e1[1].tolist())
            ex.append([ix, ix+1])
            ix += 2
            edgesy.append(nodes[i,j].e2[0].tolist())
            edgesy.append(nodes[i,j].e2[1].tolist())
            ey.append([iy, iy+1])
            iy += 2
            qx.append(q)
            qx.append(q)
            q += 1
            qy.append(q)
            qy.append(q)
            q += 1
            

    faces = []
    # Create Mesh Datablock
    mesh1 = bpy.data.meshes.new("edges x")
    mesh1.from_pydata(edgesx, ex, faces)
    mesh1.attributes.new(name="n", type="INT", domain="POINT")
    mesh1.attributes.new(name="n1", type="INT", domain="POINT")
    mesh1.attributes.new(name="n2", type="INT", domain="POINT")
    mesh1.attributes.new(name="c", type="FLOAT_VECTOR", domain="POINT")
    mesh1.attributes.new(name="a", type="BOOLEAN", domain="POINT")
    mesh1.attributes.new(name="q", type="INT", domain="POINT")
    mesh1.attributes.new(name="it", type="INT", domain="POINT")
    mesh1.attributes["n"].data.foreach_set("value",nx)
    mesh1.attributes["n1"].data.foreach_set("value",n1x)
    mesh1.attributes["n2"].data.foreach_set("value",n2x)
    mesh1.attributes["c"].data.foreach_set("vector",cx)
    mesh1.attributes["a"].data.foreach_set("value",ax)
    mesh1.attributes["q"].data.foreach_set("value",qx)
    mesh1.attributes["it"].data.foreach_set("value",itx)
    
    mesh2 = bpy.data.meshes.new("edges y")
    mesh2.from_pydata(edgesy, ey, faces)
    mesh2.attributes.new(name="n", type="INT", domain="POINT")
    mesh2.attributes.new(name="n1", type="INT", domain="POINT")
    mesh2.attributes.new(name="n2", type="INT", domain="POINT")
    mesh2.attributes.new(name="c", type="FLOAT_VECTOR", domain="POINT")
    mesh2.attributes.new(name="a", type="BOOLEAN", domain="POINT")
    mesh2.attributes.new(name="q", type="INT", domain="POINT")
    mesh2.attributes.new(name="it", type="INT", domain="POINT")
    mesh2.attributes["n"].data.foreach_set("value",ny)
    mesh2.attributes["n1"].data.foreach_set("value",n1y)
    mesh2.attributes["n2"].data.foreach_set("value",n2y)
    mesh2.attributes["c"].data.foreach_set("vector",cy)
    mesh2.attributes["a"].data.foreach_set("value",ay)
    mesh2.attributes["it"].data.foreach_set("value",ity)
    #mesh2.attributes["q"].data.foreach_set("value",qx)
        
    # Create Object and link to scene
    ob1 = bpy.data.objects.new("edges x object", mesh1)
    vs = ob1.data.vertices
    ob1.data.vertex_colors.new()
    
    #ob1.select = True
    ob2 = bpy.data.objects.new("edges y object", mesh2)
    ob2.data.vertex_colors.new()
    #ob2.select = True
    bpy.context.scene.collection.objects.link(ob1)
    bpy.context.scene.collection.objects.link(ob2)
    i = 0
    obs = [ob1, ob2]
    cols = [[.3922, .5137, 0.6196, 1],[1.0, 0.6353, 0.1451, 1]]
    cols = [[.6, .6, 1, 1],[1, .6, .6, 1],[.6, 1, .6, 1]]
    #cols = [[.6, .6, 1, 1],[1, .6, .6, 1],[.6, 1, .6, 1]]
    for ob in obs:
        makeActive(ob)
        #mergeByThreshold(ob, 0.01)
        if i == 0:            
            addGeoNodes(ob, "pipes")
        #addBevel(ob, 3, 0.1)
        colattr = bpy.context.object.data.color_attributes.new(name = "vcol", type="FLOAT_COLOR", domain='POINT')
        for v_index in range(len(bpy.context.object.data.vertices)):
            colattr.data[v_index].color = cols[i]
        i+=1
        
    coll1 = bpy.ops.collection.create(name  = "EdgeCollection")
    bpy.context.scene.collection.children.link(bpy.data.collections["EdgeCollection"])
    makeActive(ob2)
    bpy.context.scene.collection.objects.unlink(bpy.context.object)
    bpy.data.collections["EdgeCollection"].objects.link(ob2)
    bpy.data.collections["EdgeCollection"].hide_viewport=True
    makeActive(ob1)
    return ob1
    
context = bpy.context
scene = context.scene
bpy.ops.object.select_all(action='DESELECT')

for c in scene.collection.children:
    if c.name == 'EdgeCollection':
        scene.collection.children.unlink(c)

for c in bpy.data.collections:
    if c.name == 'EdgeCollection':
        if not c.users:
            bpy.data.collections.remove(c)
        
for ob in bpy.context.scene.objects:
    if ob.type == 'MESH':
        if "edges" in ob.name:
            ob.select_set(True)
    else:
        ob.select_set(False)
    bpy.ops.object.delete()

for m in bpy.data.meshes:
    if "edges" in m.name:
        m.user_clear()
        bpy.data.meshes.remove(m)

def generateCubeWireFrame():
    bpy.ops.mesh.primitive_cube_add(location =(0,0,0))
    o = bpy.data.objects['Cube']
    makeActive(o)
    bpy.ops.object.modifier_add(type='WIREFRAME')
    bpy.ops.object.material_slot_add()
    mat = bpy.data.materials.get('wire')
    o.material_slots[0].material = mat
    
def generateConfigs(c):
    configs = np.ndarray(shape=(2,2,2), dtype=np.int32)
    match c:
        # all different
        case 0:
            
            configs[0,0,0] = 0
            configs[1,0,0] = 3
            configs[0,1,0] = 5
            configs[1,1,0] = 6
            configs[0,0,1] = 7
            configs[1,0,1] = 4
            configs[0,1,1] = 2
            configs[1,1,1] = 1
        # 4 versions rotated around diagonal axis
        case 1:

            configs[0,0,0] = 5
            configs[1,0,0] = 6
            configs[0,1,0] = 0
            configs[1,1,0] = 3
            configs[0,0,1] = 3
            configs[1,0,1] = 0
            configs[0,1,1] = 6
            configs[1,1,1] = 5
        case 2:
            configs[0,0,0] = 1
            configs[1,0,0] = 2
            configs[0,1,0] = 4
            configs[1,1,0] = 7
            configs[0,0,1] = 7
            configs[1,0,1] = 4
            configs[0,1,1] = 2
            configs[1,1,1] = 1
        # 4 versions rotated around one axis
        case 3:
            configs[0,0,0] = 7
            configs[1,0,0] = 4
            configs[0,1,0] = 0
            configs[1,1,0] = 3
            configs[0,0,1] = 0
            configs[1,0,1] = 3
            configs[0,1,1] = 7
            configs[1,1,1] = 4
        case 4:
            configs[0,0,0] = 6
            configs[1,0,0] = 5
            configs[0,1,0] = 1
            configs[1,1,0] = 2
            configs[0,0,1] = 1
            configs[1,0,1] = 2
            configs[0,1,1] = 6
            configs[1,1,1] = 5
        # 2 versions 
        case 5:
            configs[0,0,0] = 7
            configs[1,0,0] = 0
            configs[0,1,0] = 0
            configs[1,1,0] = 7
            configs[0,0,1] = 0
            configs[1,0,1] = 7
            configs[0,1,1] = 7
            configs[1,1,1] = 0
        case 6:
            configs[0,0,0] = 6
            configs[1,0,0] = 1
            configs[0,1,0] = 1
            configs[1,1,0] = 6
            configs[0,0,1] = 1
            configs[1,0,1] = 6
            configs[0,1,1] = 6
            configs[1,1,1] = 1
        case 7:
            configs[0,0,0] = 5
            configs[1,0,0] = 2
            configs[0,1,0] = 2
            configs[1,1,0] = 5
            configs[0,0,1] = 2
            configs[1,0,1] = 5
            configs[0,1,1] = 5
            configs[1,1,1] = 2
        case 8:
            configs[0,0,0] = 4
            configs[1,0,0] = 3
            configs[0,1,0] = 3
            configs[1,1,0] = 4
            configs[0,0,1] = 3
            configs[1,0,1] = 4
            configs[0,1,1] = 4
            configs[1,1,1] = 3
        case '111':
            configs[0,0,0] = 0
            configs[1,0,0] = 7
            configs[0,1,0] = 7
            configs[1,1,0] = 0
            configs[0,0,1] = 7
            configs[1,0,1] = 0
            configs[0,1,1] = 0
            configs[1,1,1] = 7
        case '011':
            configs[1,0,0] = 0
            configs[0,0,0] = 3
            configs[0,1,0] = 7
            configs[1,1,0] = 4
            configs[0,0,1] = 7
            configs[1,0,1] = 4
            configs[0,1,1] = 0
            configs[1,1,1] = 3
        case '101':
            configs[1,0,0] = 0
            configs[0,0,0] = 7
            configs[0,1,0] = 5
            configs[1,1,0] = 2
            configs[0,0,1] = 7
            configs[1,0,1] = 0
            configs[0,1,1] = 2
            configs[1,1,1] = 5
        case '001':
            configs[1,0,0] = 0
            configs[0,0,0] = 3
            configs[0,1,0] = 5
            configs[1,1,0] = 6
            configs[0,0,1] = 7
            configs[1,0,1] = 0
            configs[0,1,1] = 2
            configs[1,1,1] = 3
            
    return configs
#nodes = generateRandomStructure([3,3,3], 1)
x = 5
y = 5
z = 5  
sp = 1/max(x,y,z)
fabric4 = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,0,1],
                    [0,0,1,0]])# 1/3/1
fabric5 = np.array([[0,0,0,0,1],
                    [0,0,1,0,0],
                    [1,0,0,0,0],
                    [0,0,0,1,0],
                    [0,1,0,0,0]])
fabric8 = np.array([[1,0,1,0,0,0,0,0],
                    [0,0,0,0,0,1,0,1],
                    [1,0,0,0,0,0,1,0],
                    [0,0,0,1,0,1,0,0],
                    [0,0,0,0,1,0,1,0],
                    [0,1,0,1,0,0,0,0],
                    [0,0,1,0,1,0,0,0],
                    [0,1,0,0,0,0,0,1]])   
fabric9 = np.array([[0,1,0,1,0,0,0,0,0],
                    [0,0,1,0,1,0,0,0,0],
                    [0,1,0,0,0,1,0,0,0],
                    [0,0,0,1,0,0,0,1,0],
                    [1,0,0,0,0,0,1,0,0],
                    [0,0,0,1,0,0,0,0,1],
                    [0,1,0,0,1,0,0,1,0],
                    [0,0,0,1,0,0,0,0,0],
                    [1,0,1,0,0,1,1,0,0],])

#nodes = generateFabricShiftStructure(3,[x,y,z], sp, 7, fabric8, 2, 3)

#nodes = generateABCStructure(3,[x,y,z], sp, 7, 7, 1, 3)

#nodes = generateFundamentalStructure(0,0,0, 0,0,0, sp,[x,y,z])

c = '000'

configs = generateConfigs(c)
x_0 = [0,0,0]
x_1 = [1,1,1]

#name = "plain_fund="+c+"_"+str(x)+"x"+str(y)+"x"+str(z)
name = "plain_fund="+str(x_0)+"-"+str(x_1)+"_"+str(x)+"x"+str(y)+"x"+str(z)

#name += "_exp"
#nodes = generateFundamentalStructureExplicit(configs, sp,[x,y,z], name=name)

nodes = generateFundamentalStructure(x_0[0], x_0[1], x_0[2], 
                                    x_1[0], x_1[1], x_1[2], 
                                    sp, [x,y,z], name=name)

makeActive(nodes)
bpy.ops.object.modifier_apply(modifier="GeometryNodes")
addGeoNodes(nodes, "scale")

bpy.context.view_layer.objects.active = bpy.data.objects['LineArt']
#bpy.context.object.grease_pencil_modifiers["Line Art"].source_type = 'OBJECT'
#bpy.context.object.grease_pencil_modifiers["Line Art"].source_object = bpy.data.objects[name]

#generateCubeWireFrame()
#bpy.ops.object.mode_set(mode='VERTEX_PAINT')
#bpy.context.view_layer.objects.active = bpy.context.scene.objects["Plane"]
#bpy.ops.paint.vertex_color_hsv(h=0, s=2.0, v=2.0)
#bpy.ops.paint.vertex_color_set()
