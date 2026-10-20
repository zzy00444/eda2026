// Standalone, single-thread FPGA delay estimator. All parameters are linked in.
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include "model.hpp"

extern "C" const short architecture_table[];

struct Pin { int x,y,port; };
static Pin parse_pin(std::string_view text) {
    Pin p{};
    if (text.size()<9 || text.substr(0,4)!="SRB_") throw std::runtime_error("invalid pin prefix");
    const char *begin=text.data()+4, *end=text.data()+text.size();
    auto x=std::from_chars(begin,end,p.x);
    if(x.ec!=std::errc() || x.ptr==end || *x.ptr!='_') throw std::runtime_error("invalid x coordinate");
    auto y=std::from_chars(x.ptr+1,end,p.y);
    if(y.ec!=std::errc() || y.ptr==end || *y.ptr!='/') throw std::runtime_error("invalid y coordinate");
    p.port=lookup_port(std::string_view(y.ptr+1,end-y.ptr-1));
    if(p.port<0) throw std::runtime_error("unknown port");
    if(p.x<0 || p.y<0 || p.x>10000 || p.y>10000) throw std::runtime_error("coordinate out of range");
    return p;
}
static inline int mod(int x,int n) {int r=x%n; return r<0?r+n:r;}

#if NET_CORRECTION
static inline double net_correction(const Pin& s,const Pin& t,
                                    int xlo,int xhi,int ylo,int yhi) {
    const auto& sp=ports[s.port];
    if(sp.dx==0 && sp.dy==0) return 0;
    int lx=s.x+sp.dx,ly=s.y+sp.dy;
    double wall=0;
    if(sp.dx!=0 && sp.dy==0 &&
       unsigned(lx-net_wall_left)<=unsigned(net_wall_right-net_wall_left) &&
       unsigned(s.y)<unsigned(net_wall_height) && net_wall_delays[s.y] &&
       ((s.x<net_wall_left && sp.dx>0)||(s.x>net_wall_right && sp.dx<0))) {
        lx+=(sp.dx>0 ? 1 : -1)*(net_wall_right-net_wall_left+1);
        const bool cross=(lx<net_wall_left && t.x>net_wall_right)||
                         (t.x<net_wall_left && lx>net_wall_right);
        const bool corridor=t.x>=net_wall_left && t.x<=net_wall_right;
        const int low=std::clamp(ylo,0,net_wall_height);
        const int high=std::clamp(yhi+1,0,net_wall_height);
        const bool door=net_door_prefix[high]>net_door_prefix[low];
        const int kind=cross ? (door ? 1 : 2) : (corridor ? 4 : 3);
        wall=net_wall_delays[s.y]*net_coefficients[kind];
    }
    // A landing inside the endpoint interval adds no gap cost.  Only an
    // overshoot crosses a gap twice; clamp prefix indices at the chip edges.
    int gap=0;
    if(sp.dx!=0) {
        if(lx<xlo) gap+=gap_x_prefix[std::min(xlo,gap_x_limit)]-
                         gap_x_prefix[std::clamp(lx,0,gap_x_limit)];
        else if(lx>xhi) gap+=gap_x_prefix[std::clamp(lx,0,gap_x_limit)]-
                              gap_x_prefix[std::min(xhi,gap_x_limit)];
    }
    if(sp.dy!=0) {
        if(ly<ylo) gap+=gap_y_prefix[std::min(ylo,gap_y_limit)]-
                         gap_y_prefix[std::clamp(ly,0,gap_y_limit)];
        else if(ly>yhi) gap+=gap_y_prefix[std::clamp(ly,0,gap_y_limit)]-
                              gap_y_prefix[std::min(yhi,gap_y_limit)];
    }
    return (2*gap)*net_coefficients[0]+wall;
}
#endif

static inline int estimate(const Pin& s,const Pin& t) {
    if(s.x==t.x && s.y==t.y && s.port==t.port) return 0;
    const auto &sp=ports[s.port], &tp=ports[t.port];
    int dx=t.x-s.x,dy=t.y-s.y,ax=std::abs(dx),ay=std::abs(dy);
    int xlo=std::min(s.x,t.x),xhi=std::max(s.x,t.x);
    int ylo=std::min(s.y,t.y),yhi=std::max(s.y,t.y);
    int gap_x=0,gap_y=0;
#if FAST_GAP_TABLES
    gap_x=gap_x_prefix[std::min(xhi,gap_x_limit)]-gap_x_prefix[std::min(xlo,gap_x_limit)];
    gap_y=gap_y_prefix[std::min(yhi,gap_y_limit)]-gap_y_prefix[std::min(ylo,gap_y_limit)];
#else
    for(const auto& g:gap_lines) {
        if(g.vertical) gap_x+=g.delay*(xlo<=g.site && g.site<xhi);
        else gap_y+=g.delay*(ylo<=g.site && g.site<yhi);
    }
#endif
    double baseline;
    bool supported=sp.source>=0 && tp.target>=0;
    if(supported) {
        int direction=2*(dx<0)+(dy<0),r=mod(dx,10)*12+mod(dy,12);
        size_t index=((size_t(direction)*n_sources+sp.source)*120+r)*n_targets+tp.target;
        int value=architecture_table[index];
        supported=value!=32767;
        baseline=(slopes[direction][0]*dx+slopes[direction][1]*dy+value)/12.0+gap_x+gap_y;
        if(!supported) baseline=309+12.7*ax+10.6*ay+0.93*(gap_x+gap_y);
    } else baseline=309+12.7*ax+10.6*ay+0.93*(gap_x+gap_y);
    if(!supported) return std::max(0,int(std::nearbyint(baseline)));
    model_real f[n_features]{};
    f[F_sx]=s.x;f[F_sy]=s.y;f[F_tx]=t.x;f[F_ty]=t.y;
    f[F_dx]=dx;f[F_dy]=dy;f[F_ax]=ax;f[F_ay]=ay;
    f[F_manhattan]=ax+ay;f[F_max_axis]=std::max(ax,ay);
    f[F_gap_x]=gap_x;f[F_gap_y]=gap_y;
    f[F_same_row]=s.y==t.y;f[F_same_column]=s.x==t.x;
    f[F_source_port]=sp.sp;f[F_source_family]=sp.sf;f[F_source_bit]=sp.bit;
    f[F_target_port]=tp.tp;f[F_target_family]=tp.tf;f[F_target_bit]=tp.bit;
    int cross_block_count=0;
    for(const auto& b:blocks) {
        bool row=(b.lower<=s.y && s.y<=b.upper)||(b.lower<=t.y && t.y<=b.upper);
        const bool crossed=row && xlo<b.left && xhi>b.right;
        f[F_cross_block_1+b.id-1]=crossed;
        cross_block_count+=crossed;
    }
    f[F_first_dx]=sp.dx;f[F_first_dy]=sp.dy;
    int rx=dx-sp.dx,ry=dy-sp.dy;
    f[F_remaining_x]=rx;f[F_remaining_y]=ry;
    f[F_remaining_ax]=std::abs(rx);f[F_remaining_ay]=std::abs(ry);
    f[F_dx_mod]=mod(dx,10);f[F_dy_mod]=mod(dy,12);
    f[F_ax_mod]=ax%10;f[F_ay_mod]=ay%12;
    f[F_remaining_x_mod]=mod(rx,10);f[F_remaining_y_mod]=mod(ry,12);
    f[F_remaining_ax_mod]=std::abs(rx)%10;f[F_remaining_ay_mod]=std::abs(ry)%12;
    f[F_sy_mod100]=s.y%100;f[F_ty_mod100]=t.y%100;
    f[F_architecture_delay]=baseline;
    if constexpr (n_features > 43) {
        const int source_port = sp.sp, target_port = tp.tp;
        const int source_family = sp.sf, target_family = tp.tf;
        f[F_port_pair] = category_code(F_port_pair, source_port*512 + target_port);
        f[F_family_pair] = category_code(F_family_pair, source_family*64 + target_family);
        f[F_bit_pair] = category_code(F_bit_pair, sp.bit*16 + tp.bit);
        f[F_direction] = category_code(F_direction, 2*(dx<0) + (dy<0));
        const int distance = ax + ay;
        // NumPy bool-array addition in stage3_features is logical OR, so the
        // trained category is simply whether the distance reaches ten.
        f[F_distance_bucket] = category_code(F_distance_bucket, distance>=10);
        f[F_architecture_minus_linear] = baseline -
            (309.0 + 12.7*ax + 10.6*ay + 0.93*(gap_x+gap_y));
        f[F_cross_block_count] = cross_block_count;
        f[F_source_region] = category_code(F_source_region, s.x/10);
        f[F_target_region] = category_code(F_target_region, t.x/10);
        f[F_same_region] = (s.x/10)==(t.x/10);
        f[F_axis_dominant] = ax>=ay;
    }
#if TOPOLOGY_FEATURES
    const int distance_direction=2*(dx<0)+(dy<0);
    const int cross_count=int(f[F_cross_block_count]);
    f[F_cross_block_direction]=category_code(F_cross_block_direction,cross_count*4+distance_direction);
    f[F_sy_mod50]=s.y%50; f[F_ty_mod50]=t.y%50;
    f[F_ylo_mod50]=ylo%50; f[F_yhi_mod50]=yhi%50;
    f[F_y_band]=category_code(F_y_band,ylo/100);
    f[F_same_y_band]=(s.y/100)==(t.y/100);
    f[F_xlo_to_block]=xlo-76; f[F_block_to_xhi]=89-xhi;
    f[F_x_span_block]=std::max(0,std::min(xhi,89)-std::max(xlo,76));
    f[F_near_block_left]=xlo<76 && xhi>=76;
    f[F_near_block_right]=xlo<=89 && xhi>89;
    f[F_block_y_span]=yhi-ylo;
    f[F_sy_edge50]=std::min(s.y%50,49-s.y%50);
    f[F_ty_edge50]=std::min(t.y%50,49-t.y%50);
    f[F_ylo_edge50]=std::min(ylo%50,49-ylo%50);
    f[F_yhi_edge50]=std::min(yhi%50,49-yhi%50);
    double envelope=baseline;
    const int residue=mod(dx,10)*12+mod(dy,12);
    for(int direction=0;direction<4;++direction) {
        if(direction==distance_direction) continue;
        const int penalty=(slopes[distance_direction][0]-slopes[direction][0])*dx+
                          (slopes[distance_direction][1]-slopes[direction][1])*dy;
        if(penalty>=envelope_bounds[distance_direction][direction]) continue;
        size_t index=((size_t(direction)*n_sources+sp.source)*120+residue)*n_targets+tp.target;
        int value=architecture_table[index];
        if(value!=32767) envelope=std::max(envelope,
            (slopes[direction][0]*dx+slopes[direction][1]*dy+value)/12.0+gap_x+gap_y);
    }
    f[F_architecture_envelope]=envelope;
    f[F_envelope_delta]=envelope-baseline;
#endif
    double prediction=baseline+correction(f);
#if NET_CORRECTION
    prediction+=net_correction(s,t,xlo,xhi,ylo,yhi);
#endif
    return std::max(0,int(std::nearbyint(prediction)));
}

#ifndef ESTIMATE_NO_MAIN
int main(int argc,char**argv) {
    const char *input=nullptr,*output=nullptr;
    for(int i=1;i<argc;++i) {
        if(std::strcmp(argv[i],"-in")==0 && i+1<argc) input=argv[++i];
        else if(std::strcmp(argv[i],"-out")==0 && i+1<argc) output=argv[++i];
        else {std::fprintf(stderr,"Usage: estimate -in requests.csv -out result.csv\n");return 1;}
    }
    if(!input||!output || std::strcmp(input,output)==0) {std::fprintf(stderr,"Distinct input and output paths required\n");return 1;}
    FILE *in=std::fopen(input,"rb");
    if(!in){std::perror(input);return 1;}
    FILE *out=std::fopen(output,"wb");
    if(!out){std::perror(output);std::fclose(in);return 1;}
    std::array<char,1<<20> in_buffer{},out_buffer{};
    std::setvbuf(in,in_buffer.data(),_IOFBF,in_buffer.size());
    std::setvbuf(out,out_buffer.data(),_IOFBF,out_buffer.size());
    char *line=nullptr; size_t capacity=0; ssize_t length;
    unsigned long long count=0;
    int status=0;
    try {
        if(getline(&line,&capacity,in)<0) throw std::runtime_error("missing CSV header");
        std::string_view header(line);
        if(header.substr(0,3)=="\xef\xbb\xbf") header.remove_prefix(3);
        while(!header.empty() && (header.back()=='\r'||header.back()=='\n')) header.remove_suffix(1);
        if(header!="From,To") throw std::runtime_error("expected From,To header");
        std::fputs("From,To,delay\n",out);
        while((length=getline(&line,&capacity,in))>=0) {
            while(length && (line[length-1]=='\r'||line[length-1]=='\n')) --length;
            if(!length) continue;
            std::string_view row(line,length);
            auto comma=row.find(',');
            if(comma==row.npos || row.find(',',comma+1)!=row.npos) throw std::runtime_error("expected two CSV fields");
            Pin source=parse_pin(row.substr(0,comma)),target=parse_pin(row.substr(comma+1));
            int delay=estimate(source,target);
            std::fwrite(line,1,length,out);
            char number[32];number[0]=',';
            auto end=std::to_chars(number+1,number+30,delay).ptr;
            *end++='\n';std::fwrite(number,1,end-number,out);
            ++count;
        }
        if(std::ferror(in)) throw std::runtime_error("input read error");
    } catch(const std::exception&e) {std::fprintf(stderr,"After %llu rows: %s\n",count,e.what());status=2;}
    std::free(line);
    if(std::fclose(in)!=0) status=2;
    if(std::fclose(out)!=0) {std::perror("output write");status=2;}
    std::fprintf(stderr,"Processed %llu rows\n",count);
    return status;
}
#endif
