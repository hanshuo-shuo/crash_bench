"""Prospective visible obstruction candidates; dimensions are not tuned on policy results."""
import math


def box(name, lo, hi):
    return dict(name='red_ra_'+name, lower=list(lo), upper=list(hi))


def solid_face(lower, upper, axis, side, name, thick=.012):
    lo = [v-thick for v in lower]; hi = [v+thick for v in upper]
    if side == 0: hi[axis] = lower[axis]
    else: lo[axis] = upper[axis]
    return box(name, lo, hi)


def slit(lower, upper, gap, open_roof=False):
    result = [solid_face(lower,upper,a,s,'%d_%d'%(a,s)) for a in range(3) for s in (0,1) if (a,s)!=(2,1)]
    if open_roof: return result
    roof = solid_face(lower,upper,2,1,'roof'); center=(lower[0]+upper[0])/2
    if center-gap/2 > roof['lower'][0]:
        result.append(box('roof_left',roof['lower'],[center-gap/2]+roof['upper'][1:]))
    if center+gap/2 < roof['upper'][0]:
        result.append(box('roof_right',[center+gap/2]+roof['lower'][1:],roof['upper']))
    return result


def cage(lower, upper, gap, open_roof=False, bar=.012):
    result = [solid_face(lower,upper,2,0,'floor')]
    for normal in range(3):
        for side in (0,1):
            if normal==2 and (side==0 or open_roof):continue
            template=solid_face(lower,upper,normal,side,'unused')
            varying=0 if normal!=0 else 1
            a,b=template['lower'][varying],template['upper'][varying]
            mid=(a+b)/2
            # Central gap is exactly gap; repeated bars cover both edges.
            spans=[]
            right=mid+gap/2
            while right < b:
                spans.append((right,min(right+bar,b)));right+=bar+gap
            left=mid-gap/2
            while left > a:
                spans.append((max(a,left-bar),left));left-=bar+gap
            spans += [(a,min(a+bar,b)),(max(a,b-bar),b)]
            for index,(x,y) in enumerate(spans):
                lo=template['lower'][:];hi=template['upper'][:];lo[varying]=x;hi[varying]=y
                if x<y:result.append(box('%d_%d_%d'%(normal,side,index),lo,hi))
    return result
