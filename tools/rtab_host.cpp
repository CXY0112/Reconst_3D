// Windows C-ABI host: RTAB-Map owns its embedded Python interpreter in this process.
#define NOMINMAX
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <io.h>
#include <fcntl.h>
using Create=void*(*)(const char*);
using CreateConfigured=void*(*)(const char*,int);
using OptimizerInfo=const char*(*)(void*);
using Process=int(*)(void*,const unsigned char*,const unsigned short*,int,int,const double*,double,int*);
using ProcessPose=int(*)(void*,const unsigned char*,const unsigned short*,int,int,const double*,double,const double*,double,int*);
using Poses=int(*)(void*,double*,int);
using Times=void(*)(void*,double*);
using Close=int(*)(void*);
using Error=const char*(*)();
static HANDLE channel,input;
static bool read(void* p,size_t n){char* c=(char*)p;while(n){DWORD got=0;if(!ReadFile(input,c,(DWORD)(n>1048576?1048576:n),&got,nullptr)||!got)return false;c+=got;n-=got;}return true;}
static void send(const void* p,size_t n){DWORD written;const char* c=(const char*)p;while(n){DWORD k=(DWORD)(n>1048576?1048576:n);if(!WriteFile(channel,c,k,&written,nullptr)||!written)TerminateProcess(GetCurrentProcess(),9);c+=written;n-=written;}}
template<class T> static void out(T v){send(&v,sizeof(v));}
int wmain(int argc,wchar_t** argv){
    if(argc!=4 && argc!=5)return 2;
    DuplicateHandle(GetCurrentProcess(),GetStdHandle(STD_OUTPUT_HANDLE),GetCurrentProcess(),&channel,0,FALSE,DUPLICATE_SAME_ACCESS);
    DuplicateHandle(GetCurrentProcess(),GetStdHandle(STD_INPUT_HANDLE),GetCurrentProcess(),&input,0,FALSE,DUPLICATE_SAME_ACCESS);
    _dup2(_fileno(stderr),_fileno(stdout));
    SetStdHandle(STD_OUTPUT_HANDLE,GetStdHandle(STD_ERROR_HANDLE));
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS|LOAD_LIBRARY_SEARCH_USER_DIRS);
    AddDllDirectory(argv[1]);AddDllDirectory(argv[2]);
    auto dll=LoadLibraryExW((std::wstring(argv[2])+L"\\online_bridge.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DEFAULT_DIRS|LOAD_LIBRARY_SEARCH_USER_DIRS);
    auto reply=[&](int status,int node,const std::vector<double>& poses,const double* timing,const std::string& error){out<int>(status);out<int>(node);send(timing,40);out<int>((int)poses.size()/13);if(!poses.empty())send(poses.data(),poses.size()*8);out<int>((int)error.size());if(!error.empty())send(error.data(),error.size());};
    double timing[5]={};std::vector<double> data;
    if(!dll){reply(-1,0,data,timing,"Cannot load RTAB bridge: "+std::to_string(GetLastError()));return 3;}
    auto create=(Create)GetProcAddress(dll,"online_create");auto process=(Process)GetProcAddress(dll,"online_process");auto getposes=(Poses)GetProcAddress(dll,"online_poses");auto times=(Times)GetProcAddress(dll,"online_times");auto close=(Close)GetProcAddress(dll,"online_close");auto error=(Error)GetProcAddress(dll,"online_error");
    if(!create||!process||!getposes||!times||!close||!error){reply(-1,0,data,timing,"Missing native exports");return 4;}
    auto processPose=(ProcessPose)GetProcAddress(dll,"online_process_pose");
    int bytes=WideCharToMultiByte(CP_UTF8,0,argv[3],-1,nullptr,0,nullptr,nullptr);std::string path(bytes,'\0');WideCharToMultiByte(CP_UTF8,0,argv[3],-1,path.data(),bytes,nullptr,nullptr);
    bool incremental=argc==5 && std::wstring(argv[4])==L"isam2";
    if(argc==5 && !incremental){reply(-1,0,data,timing,"Unknown optimizer");return 5;}
    auto configured=(CreateConfigured)GetProcAddress(dll,"online_create_configured");
    auto optimizerInfo=(OptimizerInfo)GetProcAddress(dll,"online_optimizer_info");
    if(incremental && (!configured || !optimizerInfo)){
        reply(-1,0,data,timing,"Native bridge has no iSAM2 support; update online_bridge.dll");return 5;
    }
    void* ctx=incremental?configured(path.c_str(),1):create(path.c_str());
    if(!ctx){reply(-1,0,data,timing,error());TerminateProcess(GetCurrentProcess(),5);}
    std::string capabilities=optimizerInfo?optimizerInfo(ctx):"{}";
    if(!capabilities.empty() && capabilities.back()=='}') {
        capabilities.pop_back();if(capabilities.size()>1)capabilities+=",";
        capabilities+=std::string("\"external_pose\":")+(processPose?"true}":"false}");
    }
    reply(1,0,data,timing,capabilities);
    auto fetch=[&](){data.resize(4096*13);int count=getposes(ctx,data.data(),4096);if(count<0){data.resize(-count*13);count=getposes(ctx,data.data(),-count);}if(count<0)throw std::runtime_error("Cannot read graph poses");data.resize(count*13);};
    try{
        int op;
        while(read(&op,4)){
            if(op==1 || op==4){
                int width,height;double k[4],stamp;
                if(!read(&width,4)||!read(&height,4)||!read(k,32)||!read(&stamp,8))break;
                if(width<=0||height<=0||(int64_t)width*height>4194304)throw std::runtime_error("Invalid RGB-D dimensions");
                size_t n=(size_t)width*height;std::vector<unsigned char> rgb(n*3);std::vector<unsigned short> depth(n);
                if(!read(rgb.data(),n*3)||!read(depth.data(),n*2))break;
                double externalPose[12],variance=.0004;
                if(op==4 && (!read(externalPose,96)||!read(&variance,8)))break;
                if(op==4 && !processPose)throw std::runtime_error("Native bridge lacks external pose support");
                int node=0;int status=op==4 ? processPose(ctx,rgb.data(),depth.data(),width,height,k,stamp,externalPose,variance,&node) : process(ctx,rgb.data(),depth.data(),width,height,k,stamp,&node);times(ctx,timing);fetch();
                // getLastLocationId() also advances for discarded low-motion nodes.
                // Publish a keyframe only if it exists in the optimized graph.
                if(status==2){bool present=false;for(size_t i=0;i<data.size();i+=13)if((int)data[i]==node){present=true;break;}if(!present){status=1;node=0;}}
                reply(status,node,data,timing,status<=0?error():"");
            }else if(op==2){fetch();reply(1,0,data,timing,"");}
            else if(op==3){fetch();int status=close(ctx);ctx=nullptr;reply(status?-1:1,0,data,timing,status?error():"");fflush(nullptr);TerminateProcess(GetCurrentProcess(),status?6:0);}
            else throw std::runtime_error("Unknown operation");
        }
    }catch(const std::exception& e){reply(-1,0,data,timing,e.what());}
    if(ctx)close(ctx);fflush(nullptr);TerminateProcess(GetCurrentProcess(),0);
}
