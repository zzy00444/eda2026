// Standalone, single-thread FPGA delay estimator. All parameters are linked in.
//
// Hot-path notes (stage 6):
//  * Input and output use explicit 4 MiB buffers; the previous getline + two
//    fwrite calls per row paid a locking stdio call twice per record.
//  * Pin parsing uses a digit loop instead of std::from_chars and returns an
//    error string instead of throwing, so the common path has no unwinding.
//  * The architecture table address is ready before the geometry features are
//    computed, so it is prefetched and its DRAM latency overlaps that work.
//    The envelope candidates are prefetched the same way; the arithmetic and
//    the order of the floating-point operations are unchanged, so predictions
//    are bit-identical to the previous implementation.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include "model.hpp"

extern "C" const short architecture_table[];

struct Pin { int x,y,port; };

// Returns nullptr on success, otherwise a static message.  Values are capped
// well inside int so absurd input cannot overflow; the original parser used
// std::from_chars and reported the same two categories of failure.
static inline bool parse_uint(const char*& p,const char* end,int& value) {
    if(p>=end) return false;
    unsigned v=0; const char* start=p;
    while(p<end) {
        const unsigned d=unsigned(*p-'0');
        if(d>9u) break;
        v=v*10u+d;
        if(v>1000000u) return false;
        ++p;
    }
    if(p==start) return false;
    value=int(v);
    return true;
}

static inline const char* parse_pin(std::string_view text,Pin& p) {
    if(text.size()<9 || text.compare(0,4,"SRB_")!=0) return "invalid pin prefix";
    const char *cur=text.data()+4, *end=text.data()+text.size();
    if(!parse_uint(cur,end,p.x) || cur>=end || *cur!='_') return "invalid x coordinate";
    ++cur;
    if(!parse_uint(cur,end,p.y) || cur>=end || *cur!='/') return "invalid y coordinate";
    ++cur;
    p.port=lookup_port(std::string_view(cur,size_t(end-cur)));
    if(p.port<0) return "unknown port";
    if(p.x<0 || p.y<0 || p.x>10000 || p.y>10000) return "coordinate out of range";
    return nullptr;
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

struct Prepared {
    int direction,residue;
    bool supported;
    size_t base_index;
#if TOPOLOGY_FEATURES
    bool envelope_needed[4];
#endif
};

// Resolve the architecture table addresses and issue the prefetches.  estimate()
// is prepare() + complete(), so a caller streaming many rows can run every
// prepare() first and give the loads hundreds of nanoseconds of slack instead
// of the few dozen instructions a single row allows.
static inline Prepared prepare(const Pin& s,const Pin& t) {
    Prepared p{};
    const auto &sp=ports[s.port], &tp=ports[t.port];
    const int dx=t.x-s.x,dy=t.y-s.y;
    p.direction=2*(dx<0)+(dy<0);
    p.residue=mod(dx,10)*12+mod(dy,12);
    p.supported=sp.source>=0 && tp.target>=0;
    p.base_index=((size_t(p.direction)*n_sources+sp.source)*120+p.residue)*n_targets+tp.target;
    if(p.supported) __builtin_prefetch(&architecture_table[p.base_index],0,3);
#if TOPOLOGY_FEATURES
    // The envelope only looks at quadrants whose spatial penalty can still
    // beat the selected one; those addresses are known now, so issue the loads
    // before the geometry work instead of after it.
    for(int other=0;other<4;++other) {
        if(other==p.direction) continue;
        const int penalty=(slopes[p.direction][0]-slopes[other][0])*dx+
                          (slopes[p.direction][1]-slopes[other][1])*dy;
        if(penalty>=envelope_bounds[p.direction][other]) continue;
        p.envelope_needed[other]=true;
        const size_t index=((size_t(other)*n_sources+sp.source)*120+p.residue)*n_targets+tp.target;
        __builtin_prefetch(&architecture_table[index],0,3);
    }
#endif
    return p;
}

#if FOLDED_MODEL
// Stage 7: the residual model is folded into the architecture table offline, so
// the per-query work is one table lookup plus the analytic linear and gap terms
// and the fitted five-term net correction.  No feature arithmetic, no tree.
__attribute__((noinline)) static int complete(const Pin& s,const Pin& t,const Prepared& p) {
    if(s.x==t.x && s.y==t.y && s.port==t.port) return 0;
    const int dx=t.x-s.x,dy=t.y-s.y;
    const int xlo=std::min(s.x,t.x),xhi=std::max(s.x,t.x);
    const int ylo=std::min(s.y,t.y),yhi=std::max(s.y,t.y);
    const int gap_x=gap_x_prefix[std::min(xhi,gap_x_limit)]-gap_x_prefix[std::min(xlo,gap_x_limit)];
    const int gap_y=gap_y_prefix[std::min(yhi,gap_y_limit)]-gap_y_prefix[std::min(ylo,gap_y_limit)];
    double prediction;
    if(p.supported) {
        const int value=architecture_table[p.base_index];
        if(value!=32767) {
            prediction=(slopes[p.direction][0]*dx+slopes[p.direction][1]*dy+value)/12.0+gap_x+gap_y;
            prediction+=net_correction(s,t,xlo,xhi,ylo,yhi);
            return std::max(0,int(std::nearbyint(prediction)));
        }
    }
    prediction=309+12.7*std::abs(dx)+10.6*std::abs(dy)+0.93*(gap_x+gap_y);
    return std::max(0,int(std::nearbyint(prediction)));
}
#else
// Deliberately not inlined: the exported single-tree model is one ~22 KB
// expression, and letting the compiler duplicate it for each buffered row both
// tripled the code size (+123 KB) and cost about 19% of the run time through
// instruction-cache misses.  One shared copy keeps the loop in L1i.
__attribute__((noinline)) static int complete(const Pin& s,const Pin& t,const Prepared& p) {
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
    const int direction=p.direction,residue=p.residue;
    bool supported=p.supported;
    const size_t base_index=p.base_index;
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
#if TOPOLOGY_FEATURES
    const int cross_count=cross_block_count;
    f[F_cross_block_direction]=category_code(F_cross_block_direction,cross_count*4+direction);
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
#endif
    // Load the prefetched architecture entries and build the baseline.
    double baseline;
    int value=0;
    if(supported) {
        value=architecture_table[base_index];
        supported=value!=32767;
        baseline=(slopes[direction][0]*dx+slopes[direction][1]*dy+value)/12.0+gap_x+gap_y;
        if(!supported) baseline=309+12.7*ax+10.6*ay+0.93*(gap_x+gap_y);
    } else baseline=309+12.7*ax+10.6*ay+0.93*(gap_x+gap_y);
    if(!supported) return std::max(0,int(std::nearbyint(baseline)));
    f[F_architecture_delay]=baseline;
    if constexpr (n_features > 43) {
        const int source_port = sp.sp, target_port = tp.tp;
        const int source_family = sp.sf, target_family = tp.tf;
        f[F_port_pair] = category_code(F_port_pair, source_port*512 + target_port);
        f[F_family_pair] = category_code(F_family_pair, source_family*64 + target_family);
        f[F_bit_pair] = category_code(F_bit_pair, sp.bit*16 + tp.bit);
        f[F_direction] = category_code(F_direction, direction);
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
    double envelope=baseline;
    for(int other=0;other<4;++other) {
        if(!p.envelope_needed[other]) continue;
        const size_t index=((size_t(other)*n_sources+sp.source)*120+residue)*n_targets+tp.target;
        int other_value=architecture_table[index];
        if(other_value!=32767) envelope=std::max(envelope,
            (slopes[other][0]*dx+slopes[other][1]*dy+other_value)/12.0+gap_x+gap_y);
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
#endif  // FOLDED_MODEL

static inline int estimate(const Pin& s,const Pin& t) {return complete(s,t,prepare(s,t));}

#ifndef ESTIMATE_NO_MAIN
namespace {
constexpr size_t kBufferBytes = size_t(1) << 22;
char input_buffer[kBufferBytes];
char output_buffer[kBufferBytes];
}  // namespace

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

    size_t in_len=0,in_pos=0;
    bool eof=false,too_long=false;
    const char* failure=nullptr;
    // Returns false only at end of input.  A row that does not fit the buffer
    // sets too_long so the caller reports a malformed input instead of looping.
    auto next_line=[&](const char*& begin,size_t& length)->bool {
        for(;;) {
            if(in_pos<in_len) {
                const char* nl=static_cast<const char*>(std::memchr(input_buffer+in_pos,'\n',in_len-in_pos));
                if(nl) {
                    begin=input_buffer+in_pos;
                    length=size_t(nl-begin);
                    in_pos=size_t(nl-input_buffer)+1;
                    return true;
                }
            }
            if(eof) {
                if(in_pos<in_len) {
                    begin=input_buffer+in_pos;
                    length=in_len-in_pos;
                    in_pos=in_len;
                    return true;
                }
                return false;
            }
            if(in_pos) {
                std::memmove(input_buffer,input_buffer+in_pos,in_len-in_pos);
                in_len-=in_pos;
                in_pos=0;
            }
            if(in_len==kBufferBytes) {too_long=true;return false;}
            const size_t got=std::fread(input_buffer+in_len,1,kBufferBytes-in_len,in);
            if(got==0) eof=true;
            in_len+=got;
        }
    };

    size_t out_len=0;
    auto flush=[&]() -> bool {
        if(!out_len) return true;
        const bool ok=std::fwrite(output_buffer,1,out_len,out)==out_len;
        out_len=0;
        return ok;
    };

    unsigned long long count=0;
    int status=0;
    // Rows are parsed and prepared (which issues the architecture prefetches)
    // in batches of eight while the previous batch is still being evaluated, so
    // every table load has a whole batch of work in front of it.  The row text
    // is copied out of the input buffer because a refill can memmove it.
    struct Row {
        Pin s,t;
        char text[256];
        size_t length;
    };
    constexpr int kBatch=8;
    Row rows[2][kBatch];
    Prepared prepared[2][kBatch];
    auto fill_batch=[&](int slot) -> int {
        if(failure) return 0;
        int n=0;
        while(n<kBatch) {
            const char *line=nullptr; size_t length=0;
            if(!next_line(line,length)) return n;
            while(length && (line[length-1]=='\r'||line[length-1]=='\n')) --length;
            if(!length) continue;
            if(length>=sizeof(rows[slot][n].text)) {failure="line longer than input buffer";return n;}
            Row& row=rows[slot][n];
            std::memcpy(row.text,line,length);
            row.length=length;
            const std::string_view text(row.text,length);
            const size_t comma=text.find(',');
            if(comma==text.npos || text.find(',',comma+1)!=text.npos) {failure="expected two CSV fields";return n;}
            if((failure=parse_pin(text.substr(0,comma),row.s))) return n;
            if((failure=parse_pin(text.substr(comma+1),row.t))) return n;
            prepared[slot][n]=prepare(row.s,row.t);
            ++n;
        }
        return n;
    };
    try {
        const char *line=nullptr; size_t length=0;
        if(!next_line(line,length)) throw std::runtime_error("missing CSV header");
        std::string_view header(line,length);
        if(header.substr(0,3)=="\xef\xbb\xbf") header.remove_prefix(3);
        while(!header.empty() && (header.back()=='\r'||header.back()=='\n')) header.remove_suffix(1);
        if(header!="From,To") throw std::runtime_error("expected From,To header");
        std::memcpy(output_buffer,"From,To,delay\n",14);
        out_len=14;
        int cur=0;
        int pending=fill_batch(0);
        while(pending>0) {
            const int next=cur^1;
            const int next_pending=fill_batch(next);
            for(int i=0;i<pending;++i) {
                const Row& row=rows[cur][i];
                const int delay=complete(row.s,row.t,prepared[cur][i]);
                if(out_len+row.length+40>kBufferBytes && !flush()) throw std::runtime_error("output write error");
                std::memcpy(output_buffer+out_len,row.text,row.length);
                out_len+=row.length;
                output_buffer[out_len++]=',';
                char number[16];
                int digits=0;
                unsigned magnitude=unsigned(delay);
                do {number[digits++]=char('0'+magnitude%10u); magnitude/=10u;} while(magnitude);
                while(digits) output_buffer[out_len++]=number[--digits];
                output_buffer[out_len++]='\n';
                ++count;
                if(out_len+64>kBufferBytes && !flush()) throw std::runtime_error("output write error");
            }
            cur=next;
            pending=next_pending;
        }
        if(failure) throw std::runtime_error(failure);
        if(too_long) throw std::runtime_error("line longer than input buffer");
        if(std::ferror(in)) throw std::runtime_error("input read error");
        if(!flush()) throw std::runtime_error("output write error");
    } catch(const std::exception&e) {std::fprintf(stderr,"After %llu rows: %s\n",count,e.what());status=2;}
    if(std::fclose(in)!=0) status=2;
    if(std::fclose(out)!=0) {std::perror("output write");status=2;}
    std::fprintf(stderr,"Processed %llu rows\n",count);
    return status;
}
#endif
