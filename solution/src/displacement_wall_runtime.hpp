// Evaluate the residual only for forced Net landings inside the wall.
#pragma once
#include "displacement_wall_model.hpp"
static inline int wmod(int a,int b) {int r=a%b;return r<0?r+b:r;}
static long wallPrediction(int sx,int sy,int tx,int ty,int source,int target,long base) {
    int ndx=netdx[source],ndy=netdy[source],lx=sx+ndx,ly=sy+ndy;
    if(lx<76 || lx>89 || WDoorDistance[ly]==0) return base;
    int dx=tx-sx,dy=ty-sy,ax=abs(dx),ay=abs(dy),rx=dx-ndx,ry=dy-ndy;
    auto cross=[](int a,int b) {return (a<=75 && b>=90)||(b<=75 && a>=90);};
    auto wall=[](int x,int y) {return x>=76 && x<=89 && WDoorDistance[y]>0;};
    int gx=vpre[max(sx,tx)]-vpre[min(sx,tx)],gy=hpre[max(sy,ty)]-hpre[min(sy,ty)];
    int ngx=vpre[max(sx,lx)]-vpre[min(sx,lx)]+vpre[max(lx,tx)]-vpre[min(lx,tx)];
    int ngy=hpre[max(sy,ly)]-hpre[min(sy,ly)]+hpre[max(ly,ty)]-hpre[min(ly,ty)];
    int32_t f[WFeatures]={sx,sy,tx,ty,dx,dy,ax,ay,ax+ay,
        source,target,WFamily[source],WFamily[target],WBit[source],WBit[target],
        ndx,ndy,rx,ry,abs(rx),abs(ry),max(abs(rx)-ax,0),max(abs(ry)-ay,0),
        wmod(dx,10),wmod(dy,12),wmod(rx,10),wmod(ry,12),sx%10,sy%12,tx%10,ty%12,
        sy/100,ty/100,sy%100,ty%100,cross(sx,tx),cross(lx,tx),wall(sx,sy),1,wall(tx,ty),
        WDoorDistance[sy],WDoorDistance[ly],WDoorDistance[ty],
        min(min(sx,tx),119-max(sx,tx)),min(min(sy,ty),549-max(sy,ty)),gx,gy,ngx,ngy,
        (int)base,(int)base-ngx-ngy,abs(ndx)+abs(ndy),2*(dx<0)+(dy<0)};
    return max(0L,(long)floor((double)base+wallDelta(f)+.5));
}
