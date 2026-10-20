#define ESTIMATE_NO_MAIN
#include "../src/main.cpp"
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    std::ifstream in(argv[1]); std::string header,line;
    std::getline(in,header);
    volatile long long checksum=0;
    while(std::getline(in,line)) {
        if(!line.empty() && line.back()=='\r') line.pop_back();
        auto comma=line.find(','); if(comma==std::string::npos) continue;
        Pin source{},target{};
        if(parse_pin(std::string_view(line.data(),comma),source)) continue;
        if(parse_pin(std::string_view(line.data()+comma+1,line.size()-comma-1),target)) continue;
        checksum += estimate(source,target);
    }
    std::fprintf(stderr,"checksum %lld\n",checksum);
}
