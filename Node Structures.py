import bpy
import numpy as np
from collections import namedtuple
import random
from dataclasses import dataclass

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
            return np.array([0.0, 0.0, -1.1])
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
    for i in range(nodes.shape[0]):
        for j in range(nodes.shape[1]): 
            for k in range(nodes.shape[2]):
                if((i+c*(j+k)) % (a+b) < a):
                    nodes[i,j,k].config = 0
                    nodes[i,j,k].a = True
                else:
                    nodes[i,j,k].config = config
                    nodes[i,j,k].a = False

def abcStructure2d(nodes, config, a, b, c):
    for i in range(nodes.shape[0]):
        for j in range(nodes.shape[1]): 
            if((i+c*(j)) % (a+b) < a):
                nodes[i,j].config = True
            else:
                nodes[i,j].config = False

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
                    nodes[i,j,k].e1[0][0] *= -1.0
                    nodes[i,j,k].e1[1][0] *= -1.0
                    nodes[i,j,k].e2[0][0] *= -1.0
                    nodes[i,j,k].e2[1][0] *= -1.0
                    nodes[i,j,k].e3[0][0] *= -1.0
                    nodes[i,j,k].e3[1][0] *= -1.0

                if (nodes[i,j,k].config & (1 << 2)):
                    nodes[i,j,k].e1[0][1] *= -1.0
                    nodes[i,j,k].e1[1][1] *= -1.0
                    nodes[i,j,k].e2[0][1] *= -1.0
                    nodes[i,j,k].e2[1][1] *= -1.0
                    nodes[i,j,k].e3[0][1] *= -1.0
                    nodes[i,j,k].e3[1][1] *= -1.0
                

                nodes[i,j,k].e1[0] += np.array([i * space - space / 4, j * space, k * space])
                nodes[i,j,k].e1[1] += np.array([i * space + space / 4, j * space, k * space])
                nodes[i,j,k].e2[0] += np.array([i * space, j * space - space / 4, k * space])
                nodes[i,j,k].e2[1] += np.array([i * space, j * space + space / 4, k * space])
                nodes[i,j,k].e3[0] += np.array([i * space, j * space, k * space - space / 4])
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

            nodes[i,j].e1[0] += np.array([i * space - space / 4, j * space, 0])
            nodes[i,j].e1[1] += np.array([i * space + space / 4, j * space, 0])
            nodes[i,j].e2[0] += np.array([i * space, j * space - space / 4, 0])
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



def generateABCStructure(dims, space, config, a, b, c):
    nodes = np.ndarray(shape=(dims[0],dims[1],dims[2]),dtype=nodeStruct)
    for i in range(nodes.shape[0]): 
        for j in range(nodes.shape[1]):
            for k in range(nodes.shape[2]):
                nodes[i,j,k] = nodeStruct(0, [0.0,0.0,0.0], [[0.0,0.0,0.1],[0.0,0.1,0.0]], [[0.1,0.0,0.1],[0.1,0.1,0.0]], [[0.0,0.1,0.1],[0.0,0.1,0.1]], False)

    abcStructure(nodes, config, a, b, c)
    updateStructure(nodes, space)
    return nodes


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
    
def generateStructureData(nodes, v = [], c = []) :
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
    n2x = []
    n1y = []
    n2y = []
    n1z = []
    n2z = []
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
                    n2x.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    n2x.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    for p in nodes[i,j,k].pos: cx.append(float(p))
                    for p in nodes[i,j,k].pos: cx.append(float(p))
                    ax.append(nodes[i,j,k].a)
                    ax.append(nodes[i,j,k].a)
                    ix += 2
                    qx.append(q)
                    qx.append(q)
                    q += 1
                if (j != 0):
                    edgesy.append(nodes[i,j - 1,k].e2[1].tolist())
                    edgesy.append(nodes[i,j,k].e2[0].tolist())
                    ey.append([iy, iy+1])
                    ny.append(k*nodes.shape[0]+i)
                    ny.append(k*nodes.shape[0]+i)
                    n1y.append(k * nodes.shape[0] * nodes.shape[1] + (j - 1) * nodes.shape[0] + i)
                    n1y.append(k * nodes.shape[0] * nodes.shape[1] + (j - 1) * nodes.shape[0] + i)
                    n2y.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    n2y.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    for p in nodes[i,j,k].pos: cy.append(float(p))
                    for p in nodes[i,j,k].pos: cy.append(float(p))
                    ay.append(nodes[i,j,k].a)
                    ay.append(nodes[i,j,k].a)
                    iy += 2
                    qy.append(q)
                    qy.append(q)
                    q += 1
                if (k != 0):
                    edgesz.append(nodes[i,j,k - 1].e3[1].tolist())
                    edgesz.append(nodes[i,j,k].e3[0].tolist())
                    ez.append([iz, iz+1])
                    nz.append(i*nodes.shape[2]+j)
                    nz.append(i*nodes.shape[2]+j)
                    n1z.append((k - 1) * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    n1z.append((k - 1) * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    n2z.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    n2z.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                    for p in nodes[i,j,k].pos: cz.append(float(p))
                    for p in nodes[i,j,k].pos: cz.append(float(p))
                    az.append(nodes[i,j,k].a)
                    az.append(nodes[i,j,k].a)
                    iz += 2
                    qz.append(q)
                    qz.append(q)
                    q += 1
                n1x.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                n1x.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                n2x.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                n2x.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                n1y.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                n1y.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                n2y.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                n2y.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                n1z.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                n1z.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                n2z.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                n2z.append(k * nodes.shape[0] * nodes.shape[1] + j * nodes.shape[0] + i)
                nx.append(k*nodes.shape[1]+j)
                nx.append(k*nodes.shape[1]+j)
                ny.append(k*nodes.shape[0]+i)
                ny.append(k*nodes.shape[0]+i)
                nz.append(i*nodes.shape[2]+j)
                nz.append(i*nodes.shape[2]+j)
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
    mesh1.attributes["n"].data.foreach_set("value",nx)
    mesh1.attributes["n1"].data.foreach_set("value",n1x)
    mesh1.attributes["n2"].data.foreach_set("value",n2x)
    mesh1.attributes["c"].data.foreach_set("vector",cx)
    mesh1.attributes["a"].data.foreach_set("value",ax)
    mesh1.attributes["q"].data.foreach_set("value",qx)
    
    mesh2 = bpy.data.meshes.new("edges y")
    mesh2.from_pydata(edgesy, ey, faces)
    mesh2.attributes.new(name="n", type="INT", domain="POINT")
    mesh2.attributes.new(name="n1", type="INT", domain="POINT")
    mesh2.attributes.new(name="n2", type="INT", domain="POINT")
    mesh2.attributes.new(name="c", type="FLOAT_VECTOR", domain="POINT")
    mesh2.attributes.new(name="a", type="BOOLEAN", domain="POINT")
    mesh2.attributes.new(name="q", type="INT", domain="POINT")
    mesh2.attributes["n"].data.foreach_set("value",ny)
    mesh2.attributes["n1"].data.foreach_set("value",n1y)
    mesh2.attributes["n2"].data.foreach_set("value",n2y)
    mesh2.attributes["c"].data.foreach_set("vector",cy)
    mesh2.attributes["a"].data.foreach_set("value",ay)
    mesh2.attributes["q"].data.foreach_set("value",qy)
        
    mesh3 = bpy.data.meshes.new("edges z")
    mesh3.from_pydata(edgesz, ez, faces)
    mesh3.attributes.new(name="n", type="INT", domain="POINT")
    mesh3.attributes.new(name="n1", type="INT", domain="POINT")
    mesh3.attributes.new(name="n2", type="INT", domain="POINT")
    mesh3.attributes.new(name="c", type="FLOAT_VECTOR", domain="POINT")
    mesh3.attributes.new(name="a", type="BOOLEAN", domain="POINT")
    mesh3.attributes.new(name="q", type="INT", domain="POINT")
    mesh3.attributes["n"].data.foreach_set("value",nz)
    mesh3.attributes["n1"].data.foreach_set("value",n1z)
    mesh3.attributes["n2"].data.foreach_set("value",n2z)
    mesh3.attributes["c"].data.foreach_set("vector",cz)
    mesh3.attributes["a"].data.foreach_set("value",az)
    mesh3.attributes["q"].data.foreach_set("value",qz)

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
    #ob3.select = True
    #bpy.ops.object.join
    bpy.context.scene.collection.objects.link(ob1)
    bpy.context.scene.collection.objects.link(ob2)
    bpy.context.scene.collection.objects.link(ob3)
    i = 0
    obs = [ob1, ob2, ob3]
    cols = [[.6, .6, 1, 1],[1, .6, .6, 1],[.6, 1, .6, 1]]
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
    makeActive(ob3)
    bpy.context.scene.collection.objects.unlink(bpy.context.object)
    bpy.data.collections["EdgeCollection"].objects.link(ob3)
    

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
                n2x.append(j * nodes.shape[0] + i)
                n2x.append(j * nodes.shape[0] + i)
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
                n1y.append((j - 1) * nodes.shape[0] + i)
                n1y.append((j - 1) * nodes.shape[0] + i)
                n2y.append(j * nodes.shape[0] + i)
                n2y.append(j * nodes.shape[0] + i)
                for p in nodes[i,j].pos: cy.append(float(p))
                for p in nodes[i,j].pos: cy.append(float(p))
                iy += 2
                qy.append(q)
                qy.append(q)
                q += 1
            n1x.append(j * nodes.shape[0] + i)
            n1x.append(j * nodes.shape[0] + i)
            n2x.append(j * nodes.shape[0] + i)
            n2x.append(j * nodes.shape[0] + i)
            n1y.append(j * nodes.shape[0] + i)
            n1y.append(j * nodes.shape[0] + i)
            n2y.append(j * nodes.shape[0] + i)
            n2y.append(j * nodes.shape[0] + i)
            nx.append(j)
            nx.append(j)
            ny.append(i)
            ny.append(i)
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
    mesh1.attributes["n"].data.foreach_set("value",nx)
    mesh1.attributes["n1"].data.foreach_set("value",n1x)
    mesh1.attributes["n2"].data.foreach_set("value",n2x)
    mesh1.attributes["c"].data.foreach_set("vector",cx)
    mesh1.attributes["a"].data.foreach_set("value",ax)
    mesh1.attributes["q"].data.foreach_set("value",qx)
    
    mesh2 = bpy.data.meshes.new("edges y")
    mesh2.from_pydata(edgesy, ey, faces)
    mesh2.attributes.new(name="n", type="INT", domain="POINT")
    mesh2.attributes.new(name="n1", type="INT", domain="POINT")
    mesh2.attributes.new(name="n2", type="INT", domain="POINT")
    mesh2.attributes.new(name="c", type="FLOAT_VECTOR", domain="POINT")
    mesh2.attributes.new(name="a", type="BOOLEAN", domain="POINT")
    mesh2.attributes.new(name="q", type="INT", domain="POINT")
    mesh2.attributes["n"].data.foreach_set("value",ny)
    mesh2.attributes["n1"].data.foreach_set("value",n1y)
    mesh2.attributes["n2"].data.foreach_set("value",n2y)
    mesh2.attributes["c"].data.foreach_set("vector",cy)
    mesh2.attributes["a"].data.foreach_set("value",ay)
    mesh2.attributes["q"].data.foreach_set("value",qx)
        
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
    cols = [[.6, .6, 1, 1],[1, .6, .6, 1]]
    for ob in obs:
        makeActive(ob)
        #mergeByThreshold(ob, 0.01)            
        addGeoNodes(ob, "pipes")
        #addBevel(ob, 3, 0.1)
        colattr = bpy.context.object.data.color_attributes.new(name = "vcol", type="FLOAT_COLOR", domain='POINT')
        for v_index in range(len(bpy.context.object.data.vertices)):
            colattr.data[v_index].color = cols[i]
        i+=1
context = bpy.context
scene = context.scene

for c in scene.collection.children:
    scene.collection.children.unlink(c)

for c in bpy.data.collections:
    if not c.users:
        bpy.data.collections.remove(c)
        
for ob in bpy.context.scene.objects:
    if ob.type == 'MESH':
        ob.select_set(True)
    else:
        ob.select_set(False)
    bpy.ops.object.delete()

for m in bpy.data.meshes:
    m.user_clear()
    bpy.data.meshes.remove(m)


#nodes = generateRandomStructure([3,3,3], 1)
nodes = generateABCStructure([4,4,4], .5, 7, 2, 2, 1)

generateStructureData(nodes)
#bpy.ops.object.mode_set(mode='VERTEX_PAINT')
#bpy.context.view_layer.objects.active = bpy.context.scene.objects["Plane"]
#bpy.ops.paint.vertex_color_hsv(h=0, s=2.0, v=2.0)
#bpy.ops.paint.vertex_color_set()
