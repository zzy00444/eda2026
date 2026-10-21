// Optional short-displacement correction. Its factors use generated distances.
#pragma once
static int LRank=0;
static vector<int32_t> LMap;
static vector<float> LU;
static vector<int16_t> LV;

static bool localInit() {
    if (!TB_EMB || TB_OFF[0]<32) return false;
    FILE* f=fopen(TB_SELF.c_str(),"rb");
    if (!f) return false;
    unsigned char descriptor[32];
    if (fseek(f,(long)(TB_OFF[0]-32),SEEK_SET) || fread(descriptor,1,32,f)!=32) {fclose(f);return false;}
    uint64_t offset,size,reserved;
    memcpy(&offset,descriptor+8,8);memcpy(&size,descriptor+16,8);memcpy(&reserved,descriptor+24,8);
    if (memcmp(descriptor,"PGLCF001",8) || reserved || offset>=TB_OFF[0]-32 ||
        size!=TB_OFF[0]-32-offset || size>100000000) {fclose(f);return false;}
    unsigned char magic[8];uint32_t h[6];
    if (fseek(f,(long)offset,SEEK_SET) || fread(magic,1,8,f)!=8 || fread(h,4,6,f)!=6 ||
        memcmp(magic,"PGLOC001",8) || h[0]!=(unsigned)(nSrc*nTgt) || !h[1] || h[1]>512 ||
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
    return col<0 ? 0 : estimateRound((double)dotProduct(&LU[(size_t)pair*LRank],&LV[(size_t)col*LRank],LRank));
}
