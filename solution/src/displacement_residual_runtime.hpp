// Stage-9 envelope and residual runtime. Included after the base model globals.
#pragma once
#include <zlib.h>
static vector<int16_t> RBoundTable; // [source,target,phase,direction], four adjacent values
#include "displacement_residual_model.hpp"

static bool residualInit() {
    if (!TB_EMB || TB_OFF[0] < 32) return false;
    FILE* f = fopen(TB_SELF.c_str(), "rb");
    if (!f) return false;
    unsigned char descriptor[32];
    if (fseek(f, (long)(TB_OFF[0]-32), SEEK_SET) || fread(descriptor,1,32,f)!=32) { fclose(f); return false; }
    uint64_t offset, size, rawSize;
    memcpy(&offset,descriptor+8,8); memcpy(&size,descriptor+16,8); memcpy(&rawSize,descriptor+24,8);
    if (memcmp(descriptor,"PGBND001",8) || offset>=TB_OFF[0]-32 || size!=TB_OFF[0]-32-offset ||
        size>100000000 || rawSize!=RDecodedBytes) { fclose(f); return false; }
    vector<unsigned char> packed(size), decoded(rawSize);
    if (fseek(f,(long)offset,SEEK_SET) || fread(packed.data(),1,size,f)!=size) { fclose(f); return false; }
    fclose(f);
    uLongf decodedBytes=decoded.size();
    if (uncompress(decoded.data(),&decodedBytes,packed.data(),packed.size())!=Z_OK || decodedBytes!=decoded.size()) return false;
    RBoundTable.resize((size_t)RSourceCount*RTargetCount*120*4);
    size_t at=0;
    for (int d=0;d<4;d++) for (int pair=0;pair<RSourceCount*RTargetCount;pair++) {
        int value=0;
        for (int r=0;r<120;r++) {
            if (at>=decoded.size()) return false;
            unsigned char c=decoded[at++];
            int delta;
            if (c==128) {
                if (at+2>decoded.size()) return false;
                int16_t v; memcpy(&v,decoded.data()+at,2); at+=2; delta=v;
            } else delta=(int8_t)c;
            value+=delta;
            if (value<-32768 || value>32767) return false;
            RBoundTable[((size_t)pair*120+r)*4+d]=(int16_t)value;
        }
    }
    return at==decoded.size() && crc32(0,(const Bytef*)RBoundTable.data(),RBoundTable.size()*2)==RTableCRC;
}

static inline int rmod(int value,int period) { int r=value%period; return r<0?r+period:r; }

static bool residualFeatures(int sx,int sy,int tx,int ty,int source,int target,long base,RFeature* f) {
    int sp=RSourceIndex[source],tp=RTargetIndex[target];
    if (sp<0 || tp<0) return false;
    int dx=tx-sx,dy=ty-sy,ax=abs(dx),ay=abs(dy),ndx=netdx[source],ndy=netdy[source];
    int lx=max(0,min(W-1,sx+ndx)),ly=max(0,min(H-1,sy+ndy)),rx=dx-ndx,ry=dy-ndy;
    auto cross=[](int a,int b) {return (a<=75 && b>=90)||(b<=75 && a>=90);};
    auto wall=[](int x,int y) {return x>=76 && x<=89 && RDoorDistance[y]>0;};
    int gapx=vpre[max(sx,tx)]-vpre[min(sx,tx)],gapy=hpre[max(sy,ty)]-hpre[min(sy,ty)];
    int ngx=vpre[max(sx,lx)]-vpre[min(sx,lx)]+vpre[max(lx,tx)]-vpre[min(lx,tx)];
    int ngy=hpre[max(sy,ly)]-hpre[min(sy,ly)]+hpre[max(ly,ty)]-hpre[min(ly,ty)];
    int mx=rmod(dx,10),my=rmod(dy,12);
    const int16_t* q=&RBoundTable[(((size_t)sp*RTargetCount+tp)*120+mx*12+my)*4];
    int bound=INT_MIN;
    for(int d=0;d<4;d++) bound=max(bound,(int)q[d]+(RSlopes[d][0]*(dx-mx)+RSlopes[d][1]*(dy-my))/12);
    bound+=ngx+ngy;
    f[RF_sx]=sx;f[RF_sy]=sy;f[RF_tx]=tx;f[RF_ty]=ty;
    f[RF_dx]=dx;f[RF_dy]=dy;f[RF_ax]=ax;f[RF_ay]=ay;f[RF_distance]=ax+ay;
    f[RF_source]=source;f[RF_target]=target;f[RF_source_family]=RFamily[source];f[RF_target_family]=RFamily[target];
    f[RF_source_bit]=RBit[source];f[RF_target_bit]=RBit[target];f[RF_ndx]=ndx;f[RF_ndy]=ndy;
    f[RF_rx]=rx;f[RF_ry]=ry;f[RF_rax]=abs(rx);f[RF_ray]=abs(ry);
    f[RF_overx]=max(abs(rx)-ax,0);f[RF_overy]=max(abs(ry)-ay,0);
    f[RF_dx_mod]=mx;f[RF_dy_mod]=my;f[RF_rx_mod]=rmod(rx,10);f[RF_ry_mod]=rmod(ry,12);
    f[RF_sx_mod]=sx%10;f[RF_sy_mod]=sy%12;f[RF_tx_mod]=tx%10;f[RF_ty_mod]=ty%12;
    f[RF_source_band]=sy/100;f[RF_target_band]=ty/100;f[RF_source_phase]=sy%100;f[RF_target_phase]=ty%100;
    f[RF_cross]=cross(sx,tx);f[RF_landing_cross]=cross(lx,tx);
    f[RF_source_wall]=wall(sx,sy);f[RF_landing_wall]=wall(lx,ly);f[RF_target_wall]=wall(tx,ty);
    f[RF_source_door_distance]=RDoorDistance[sy];f[RF_landing_door_distance]=RDoorDistance[ly];f[RF_target_door_distance]=RDoorDistance[ty];
    f[RF_edge_x]=min(min(sx,tx),119-max(sx,tx));f[RF_edge_y]=min(min(sy,ty),549-max(sy,ty));
    f[RF_gap_x]=gapx;f[RF_gap_y]=gapy;f[RF_net_gap_x]=ngx;f[RF_net_gap_y]=ngy;
    f[RF_base]=base;f[RF_base_minus_gap]=base-ngx-ngy;f[RF_net_length]=abs(ndx)+abs(ndy);
    f[RF_direction]=2*(dx<0)+(dy<0);f[RF_bound]=bound;f[RF_base_minus_bound]=base-bound;
#ifdef EST_RESIDUAL_PREPARE
    rPrepareFeature(f);
#endif
    return true;
}

static double residualPrediction(int sx,int sy,int tx,int ty,int source,int target,long base) {
#ifdef EST_RESIDUAL_QUICK
    double quick;
    if(rQuickPredict(sx,sy,tx,ty,source,target,base,quick)) return quick;
#endif
    RFeature f[RFeatures];
    if(!residualFeatures(sx,sy,tx,ty,source,target,base,f)) return base;
#ifdef EST_RESIDUAL_DIRECT
    double delta=rDirectPredict(f);
#else
    double delta=0;
    for(int t=0;t<RTrees;t++) {
        int at=RRoots[t];
        while(RNodes[at].left>=0) {
            const RNode& node=RNodes[at];
            bool left;
            if(node.count) {
                unsigned cat=(unsigned)(int)f[node.feature];
                left=cat/64<node.count && ((RCategories[node.cb+cat/64]>>(cat&63))&1ull);
            } else left=f[node.feature]<=node.value;
            at=left?node.left:node.right;
        }
        delta+=RNodes[at].value;
    }
#endif
    double pred=max(0.,floor((double)base+delta+.5));
    return RClamp?max(pred,(double)f[RF_bound]):pred;
}
