// Optional short-displacement correction. Its factors use generated distances.
#pragma once
static int LRank=0;
static vector<int32_t> LMap;
static vector<float> LU;
static vector<int16_t> LV;
struct LocalPart { int rank; bool direct=false; vector<float> u; vector<int16_t> v; vector<int32_t> pairmap; };
static vector<LocalPart> LParts;
static vector<int8_t> LPartOf;
#ifdef EST_LOCAL_XZ
// Stable liblzma buffer API. The allocator pointer is null (default allocator).
// The official image provides liblzma.so.5 but does not ship development headers.
extern "C" int lzma_stream_buffer_decode(uint64_t*,uint32_t,const void*,const uint8_t*,size_t*,size_t,uint8_t*,size_t*,size_t);
#endif

static bool localInit() {
    if (!TB_EMB || TB_OFF[0]<32) return false;
    FILE* f=fopen(TB_SELF.c_str(),"rb");
    if (!f) return false;
    unsigned char descriptor[32];
    if (fseek(f,(long)(TB_OFF[0]-32),SEEK_SET) || fread(descriptor,1,32,f)!=32) {fclose(f);return false;}
    uint64_t offset,size,reserved;
    memcpy(&offset,descriptor+8,8);memcpy(&size,descriptor+16,8);memcpy(&reserved,descriptor+24,8);
    bool compressed=!memcmp(descriptor,"PGLCZ001",8);
    if ((!compressed && (memcmp(descriptor,"PGLCF001",8) || reserved)) || offset>=TB_OFF[0]-32 ||
        size!=TB_OFF[0]-32-offset || size>100000000) {fclose(f);return false;}
    vector<uint8_t> decoded;
    if(compressed) {
#ifdef EST_LOCAL_XZ
        if(!reserved || reserved>100000000) {fclose(f);return false;}
        vector<uint8_t> encoded(size);decoded.resize(reserved);
        if(fseek(f,(long)offset,SEEK_SET) || fread(encoded.data(),1,size,f)!=size) {fclose(f);return false;}
        uint64_t limit=128ull<<20;size_t ip=0,op=0;
        int status=lzma_stream_buffer_decode(&limit,0,nullptr,encoded.data(),&ip,encoded.size(),decoded.data(),&op,decoded.size());
        fclose(f);
        if(status || ip!=encoded.size() || op!=decoded.size()) return false;
        f=fmemopen(decoded.data(),decoded.size(),"rb");
        if(!f) return false;
        offset=0;size=decoded.size();
#else
        fclose(f);return false;
#endif
    }
    unsigned char magic[8];uint32_t h[6];
    if (fseek(f,(long)offset,SEEK_SET) || fread(magic,1,8,f)!=8 || fread(h,4,6,f)!=6) {fclose(f);return false;}
    bool direct=!memcmp(magic,"PGLOC004",8);
    bool mixed=!memcmp(magic,"PGLOC005",8);
    bool sparse=mixed || direct || !memcmp(magic,"PGLOC003",8);
    if (sparse || !memcmp(magic,"PGLOC002",8)) {
        if(h[0]!=(unsigned)(nSrc*nTgt) || !h[1] || h[1]>32 || !h[2] || h[2]>(unsigned)NC ||
           h[3]!=(unsigned)NC || h[4]>1 || h[5]) {fclose(f);return false;}
        LMap.resize(NC);LPartOf.assign(NC,-1);LParts.resize(h[1]);
        if(fread(LMap.data(),4,NC,f)!=(size_t)NC) {fclose(f);return false;}
        size_t begin=0;
        for(unsigned part=0;part<h[1];part++) {
            uint32_t ph[4];
            if(fread(ph,4,4,f)!=4) {fclose(f);return false;}
            bool partDirect=mixed ? (bool)(ph[0]>>31) : direct;
            if(mixed) ph[0]&=0x7fffffffu;
            if(!ph[0] || ph[0]>512 || !ph[1] || begin+ph[1]>h[2] ||
               (ph[2]!=8 && ph[2]!=16) || (sparse ? (!ph[3] || ph[3]>h[0]) : ph[3]!=0)) {fclose(f);return false;}
            auto& p=LParts[part];p.rank=ph[0];p.direct=partDirect;vector<float> us(p.rank),vs(p.rank);
            size_t pairs=sparse ? ph[3] : h[0];
            if(sparse) {
                p.pairmap.resize(h[0]);
                if(fread(p.pairmap.data(),4,h[0],f)!=h[0]) {fclose(f);return false;}
                for(int value:p.pairmap) if(value < -1 || value>=(int)pairs) {fclose(f);return false;}
            }
            if(fread(us.data(),4,p.rank,f)!=(size_t)p.rank || fread(vs.data(),4,p.rank,f)!=(size_t)p.rank) {fclose(f);return false;}
            p.u.resize(pairs*p.rank);p.v.resize((size_t)ph[1]*p.rank);
            auto decode=[&](auto q) {for(size_t i=0;i<p.u.size();i++) p.u[i]=((float)q[i]*us[i%p.rank])*vs[i%p.rank];};
            if(ph[2]==8) {
                vector<int8_t> q(p.u.size());
                if(fread(q.data(),1,q.size(),f)!=q.size()) {fclose(f);return false;}
                decode(q.data());
            } else {
                vector<int16_t> q(p.u.size());
                if(fread(q.data(),2,q.size(),f)!=q.size()) {fclose(f);return false;}
                decode(q.data());
            }
            if(fread(p.v.data(),2,p.v.size(),f)!=p.v.size()) {fclose(f);return false;}
            if(h[4]) {
                // Two byte planes, then rank-major deltas at a lag of 12 columns.
                vector<int16_t> packed;packed.swap(p.v);p.v.resize(packed.size());
                const uint8_t* bytes=(const uint8_t*)packed.data();size_t elements=packed.size();
                for(size_t k=0;k<(size_t)p.rank;k++) for(size_t col=0;col<ph[1];col++) {
                    size_t source=k*ph[1]+col;uint16_t value=(uint16_t)bytes[source]|((uint16_t)bytes[elements+source]<<8);
                    if(col>=12) value+=(uint16_t)p.v[(col-12)*p.rank+k];
                    p.v[col*p.rank+k]=(int16_t)value;
                }
            }
            // Delay remapping until all global column ranges have been assigned.
            for(int row=0;row<NC;row++) if(LMap[row]>=(int)begin && LMap[row]<(int)(begin+ph[1])) LPartOf[row]=part;
            begin+=ph[1];
        }
        bool ok=begin==h[2] && (uint64_t)ftell(f)==offset+size;fclose(f);
        vector<int> starts(LParts.size());
        for(size_t i=1;i<LParts.size();i++) starts[i]=starts[i-1]+LParts[i-1].v.size()/LParts[i-1].rank;
        for(int row=0;row<NC;row++) {
            if(LMap[row]<-1 || LMap[row]>=(int)h[2] || (LMap[row]>=0 && LPartOf[row]<0)) return false;
            if(LPartOf[row]>=0) LMap[row]-=starts[LPartOf[row]];
        }
        return ok;
    }
    if (memcmp(magic,"PGLOC001",8) || h[0]!=(unsigned)(nSrc*nTgt) || !h[1] || h[1]>512 ||
        !h[2] || h[2]>(unsigned)NC || h[3]!=(unsigned)NC || (h[4]!=8 && h[4]!=16) || h[5]) {fclose(f);return false;}
    LRank=h[1];size_t pairs=h[0],cols=h[2];
    if (size!=32+4ull*NC+8ull*LRank+pairs*LRank*(h[4]/8)+2ull*cols*LRank) {fclose(f);return false;}
    LMap.resize(NC);vector<float> us(LRank),vs(LRank);
    if (fread(LMap.data(),4,NC,f)!=(size_t)NC || fread(us.data(),4,LRank,f)!=(size_t)LRank ||
        fread(vs.data(),4,LRank,f)!=(size_t)LRank) {fclose(f);return false;}
    for (int value:LMap) if (value < -1 || value>=(int)cols) {fclose(f);return false;}
    LU.resize(pairs*LRank);LV.resize(cols*LRank);
    auto decode=[&](auto value) {for(size_t i=0;i<LU.size();i++) LU[i]=((float)value[i]*us[i%LRank])*vs[i%LRank];};
    if (h[4]==8) {
        vector<int8_t> q(LU.size());
        if(fread(q.data(),1,q.size(),f)!=q.size()){fclose(f);return false;}
        decode(q.data());
    } else {
        vector<int16_t> q(LU.size());
        if(fread(q.data(),2,q.size(),f)!=q.size()){fclose(f);return false;}
        decode(q.data());
    }
    bool ok=fread(LV.data(),2,LV.size(),f)==LV.size();fclose(f);return ok;
}

static inline long localCorrection(unsigned row,unsigned pair) {
    int col=LMap[row];
    if(!LParts.empty()) {
        if(col<0) return 0;
        const auto& p=LParts[(unsigned)LPartOf[row]];
        if(!p.pairmap.empty()) {
            int selected=p.pairmap[pair];
            if(selected<0) return 0;
            pair=selected;
        }
        return estimateRound((double)dotProduct(&p.u[(size_t)pair*p.rank],&p.v[(size_t)col*p.rank],p.rank));
    }
    return col<0 ? 0 : estimateRound((double)dotProduct(&LU[(size_t)pair*LRank],&LV[(size_t)col*LRank],LRank));
}

static inline long localEstimate(unsigned row,unsigned pair,const float* baseu,int mrow) {
    if(!LParts.empty() && LMap[row]>=0) {
        const auto& p=LParts[(unsigned)LPartOf[row]];
        if(p.direct) {
            int selected=p.pairmap.empty() ? (int)pair : p.pairmap[pair];
            if(selected>=0) return estimateRound((double)dotProduct(&p.u[(size_t)selected*p.rank],&p.v[(size_t)LMap[row]*p.rank],p.rank));
        }
    }
    long base=estimateRound((double)mrow+(double)dotProduct(baseu,&Vt[(size_t)row*RANK],RANK));
    return base+localCorrection(row,pair);
}

static inline bool localPrefetch(unsigned row,unsigned pair) {
    if(LParts.empty() || LMap[row]<0) return false;
    const auto& p=LParts[(unsigned)LPartOf[row]];
    int selected=p.pairmap.empty() ? (int)pair : p.pairmap[pair];
    if(selected<0) return false;
    const float* u=&p.u[(size_t)selected*p.rank];pfT0(u);
    if(p.rank>16) pfT0(u+16);
    pfT0(&p.v[(size_t)LMap[row]*p.rank]);
    return p.direct;
}
