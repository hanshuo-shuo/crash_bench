"""Simulator-native visual grounding; decorative geoms cannot affect contacts."""
import itertools
import xml.etree.ElementTree as ET

STYLE = 'clear_blue_panels_fixed_green_goal_v1'


def edges(world, prefix, lower, upper, rgba, width=.0012):
    for axis in range(3):
        others=[i for i in range(3) if i!=axis]
        for index,sides in enumerate(itertools.product((0,1),repeat=2)):
            center=[(a+b)/2 for a,b in zip(lower,upper)];half=[width]*3
            half[axis]=(upper[axis]-lower[axis])/2
            for i,side in zip(others,sides):center[i]=(lower,upper)[side][i]
            ET.SubElement(world,'geom',dict(name='%s_%d_%d'%(prefix,axis,index),type='box',
                pos=' '.join(format(x,'.17g') for x in center),size=' '.join(format(x,'.17g') for x in half),
                rgba=rgba,contype='0',conaffinity='0',group='1'))


def ground(xml,goal_bounds):
    tree=ET.fromstring(xml);world=tree.find('worldbody')
    for geom in list(world.findall('geom')):
        if not geom.get('name','').startswith('fd_wall_'):continue
        # Alpha/color are rendering attributes only. Leave collision geometry,
        # placement, friction, solver parameters and all existing bodies intact.
        geom.set('rgba','.12 .42 .85 .12')
        center=list(map(float,geom.get('pos').split()));half=list(map(float,geom.get('size').split()))
        edges(world,'fd_visual_'+geom.get('name'),[p-s for p,s in zip(center,half)],
              [p+s for p,s in zip(center,half)],'.08 .25 .75 1',width=.0007)
    edges(world,'fd_visual_fixed_goal',goal_bounds[0],goal_bounds[1],'.1 .75 .25 1')
    return ET.tostring(tree,encoding='unicode')
