#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/Link.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/utilite/ULogger.h>
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <cstring>
#include <chrono>
#include <stdexcept>

struct OnlineContext {
    std::unique_ptr<rtabmap::Odometry> odometry;
    rtabmap::Rtabmap map;
    std::string error;
    double lastKeyframe = -1;
    int lastId = 0;
    bool closed = false;
    std::string database;
    double timings[5] = {};
    bool snapshotMode = false;
    bool geometryGraph = false;
    std::string optimizerInfo;
    std::multimap<int,rtabmap::Link> geometryLinks;
};
static thread_local std::string lastError;
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

API void * online_create_configured(const char * database, int incremental) {
    try {
        cv::setNumThreads(1);
        ULogger::setLevel(ULogger::kWarning);
        auto ctx = std::make_unique<OnlineContext>();
        rtabmap::ParametersMap params;
        params["RGBD/Enabled"] = "true";
        params["Mem/IncrementalMemory"] = "true";
        params["RGBD/CreateOccupancyGrid"] = "false";
        params["RGBD/LinearUpdate"] = "0.03";
        params["RGBD/AngularUpdate"] = "0.03";
        params["Vis/FeatureType"] = "8";
        params["Kp/DetectorStrategy"] = "8";
        params["Vis/MaxFeatures"] = "1000";
        params["Kp/MaxFeatures"] = "1000";
        params["Odom/ImageDecimation"] = "1";
        params["Odom/Strategy"] = "0"; // frame to local feature map
        params["Reg/Strategy"] = "0";  // RGB-D visual registration, not dense ICP
        if(incremental) {
            if(!rtabmap::Optimizer::isAvailable(rtabmap::Optimizer::kTypeGTSAM))
                throw std::runtime_error("iSAM2 requires a GTSAM-enabled RTAB-Map runtime; no fallback was used");
            params["Optimizer/Strategy"] = "2";
            params["GTSAM/Incremental"] = "true";
            params["GTSAM/Optimizer"] = "1";
            params["GTSAM/IncRelinearizeThreshold"] = "0.01";
            params["GTSAM/IncRelinearizeSkip"] = "1";
        }
        const auto & defaults=rtabmap::Parameters::getDefaultParameters();
        int strategy=std::stoi(defaults.at("Optimizer/Strategy"));
        ctx->optimizerInfo=incremental ?
            "{\"requested\":\"isam2\",\"strategy\":2,\"incremental\":true,\"verified\":true,\"relinearize_threshold\":0.01,\"relinearize_skip\":1}" :
            "{\"requested\":\"default\",\"strategy\":"+std::to_string(strategy)+",\"incremental\":"+
            (strategy==2 && defaults.at("GTSAM/Incremental")=="true" ? "true" : "false")+",\"verified\":true}";
        ctx->odometry.reset(rtabmap::Odometry::create(params));
        ctx->map.init(params, database);
        ctx->database = database;
        lastError.clear();
        return ctx.release();
    } catch(const std::exception & e) { lastError = e.what(); return nullptr; }
}
API void * online_create(const char * database) { return online_create_configured(database,0); }
API const char * online_optimizer_info(void * context) {
    return static_cast<OnlineContext *>(context)->optimizerInfo.c_str();
}

// 0: tracking lost, 1: tracked, 2: new keyframe. All calls made on one worker thread.
API int online_process(void * context, const unsigned char * rgb, const unsigned short * depth,
                       int width, int height, const double * intrinsics, double stamp, int * nodeId) {
    auto & ctx = *static_cast<OnlineContext *>(context);
    try {
        cv::Mat color(height, width, CV_8UC3, const_cast<unsigned char *>(rgb));
        cv::Mat dep(height, width, CV_16UC1, const_cast<unsigned short *>(depth));
        rtabmap::CameraModel model("hp60c", intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3],
                                  rtabmap::CameraModel::opticalRotation(), 0, cv::Size(width,height));
        rtabmap::SensorData data(color.clone(), dep.clone(), model, 0, stamp);
        rtabmap::OdometryInfo info;
        auto started = std::chrono::steady_clock::now();
        auto pose = ctx.odometry->process(data, &info);
        ctx.timings[0] = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        ctx.timings[1] = 0;
        ctx.timings[2] = info.reg.inliers;
        ctx.timings[3] = info.features;
        ctx.timings[4] = info.reg.matches;
        if(pose.isNull()) {
            lastError = info.reg.rejectedMsg + " features=" + std::to_string(info.features) + " matches=" + std::to_string(info.reg.matches);
            return 0;
        }
        if(ctx.lastKeyframe < 0 || stamp - ctx.lastKeyframe >= .5) {
            started = std::chrono::steady_clock::now();
            ctx.map.process(data, pose, info.reg.covariance);
            ctx.timings[1] = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            ctx.lastKeyframe = stamp;
            int id = ctx.map.getLastLocationId();
            if(id > 0 && id != ctx.lastId) {
                ctx.lastId = id;
                *nodeId = id;
                return 2;
            }
        }
        return 1;
    } catch(const std::exception & e) { lastError = e.what(); return -1; }
}

// External geometry frontend: optical-camera-to-world pose, metres. RGB is
// still passed to RTAB-Map for its existing place recognition and loop checks.
// This never calls the visual odometry object.
API int online_process_pose(void * context, const unsigned char * rgb, const unsigned short * depth,
                           int width, int height, const double * intrinsics, double stamp,
                           const double * worldPose, double variance, int * nodeId) {
    auto & ctx = *static_cast<OnlineContext *>(context);
    try {
        cv::Mat color(height,width,CV_8UC3,const_cast<unsigned char *>(rgb));
        cv::Mat dep(height,width,CV_16UC1,const_cast<unsigned short *>(depth));
        rtabmap::CameraModel model("hp60c",intrinsics[0],intrinsics[1],intrinsics[2],intrinsics[3],
                                  rtabmap::CameraModel::opticalRotation(),0,cv::Size(width,height));
        rtabmap::SensorData data(color.clone(),dep.clone(),model,0,stamp);
        rtabmap::Transform opticalWorld(worldPose[0],worldPose[1],worldPose[2],worldPose[3],
                                        worldPose[4],worldPose[5],worldPose[6],worldPose[7],
                                        worldPose[8],worldPose[9],worldPose[10],worldPose[11]);
        auto robotWorld=opticalWorld*rtabmap::CameraModel::opticalRotation().inverse();
        std::memset(ctx.timings,0,sizeof(ctx.timings));
        if(ctx.lastKeyframe<0 || stamp-ctx.lastKeyframe>=.5) {
            auto started=std::chrono::steady_clock::now();
            // Separate translation/rotation variances; conservative configured
            // values, not a calibrated covariance estimate from BUFFER-X.
            ctx.map.process(data,robotWorld,float(variance),float(.0004));
            ctx.timings[1]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            ctx.lastKeyframe=stamp;
            int id=ctx.map.getLastLocationId();
            if(id>0 && id!=ctx.lastId){ctx.lastId=id;*nodeId=id;return 2;}
        }
        return 1;
    } catch(const std::exception & e){lastError=e.what();return -1;}
}

// Each record: id plus a 3x4 camera-optical-to-world pose, with metres translation.
API int online_poses(void * context, double * output, int capacity) {
    auto & ctx = *static_cast<OnlineContext *>(context);
    const auto & poses = ctx.map.getLocalOptimizedPoses();
    if(capacity < static_cast<int>(poses.size())) return -static_cast<int>(poses.size());
    int count = 0;
    for(const auto & entry : poses) {
        output[count*13] = entry.first;
        auto pose = entry.second * rtabmap::CameraModel::opticalRotation();
        for(int i=0;i<12;++i) output[count*13+i+1] = pose[i];
        ++count;
    }
    return count;
}

API int online_close(void * context) {
    auto ctx = static_cast<OnlineContext *>(context);
    try {
        auto poses = ctx->map.getLocalOptimizedPoses();
        ctx->map.close(true);
        // Save the current optimized graph, including valid single-node scans.
        std::unique_ptr<rtabmap::DBDriver> db(rtabmap::DBDriver::create());
        if(!db->openConnection(ctx->database)) throw std::runtime_error("Cannot reopen map database");
        if(ctx->geometryGraph) {
            std::multimap<int,rtabmap::Link> previous;
            db->getAllLinks(previous);
            for(const auto & entry : previous) db->removeLink(entry.second.from(),entry.second.to());
            for(const auto & entry : ctx->geometryLinks) {
                db->addLink(entry.second);
                db->addLink(entry.second.inverse());
            }
        }
        db->saveOptimizedPoses(poses, rtabmap::Transform::getIdentity());
        db->closeConnection();
        ctx->closed = true;
        delete ctx;
        return 0;
    }
    catch(const std::exception & e) { lastError = e.what(); delete ctx; return -1; }
}
API const char * online_error() { return lastError.c_str(); }
API void online_times(void * context, double * output) {
    std::memcpy(output, static_cast<OnlineContext *>(context)->timings, sizeof(double)*5);
}

// Add an independently registered RGB-D capture. Input pose maps optical camera
// coordinates to robot world; it does not depend on previous capture timestamps.
API int snapshot_add(void * context, const unsigned char * rgb, const unsigned short * depth,
                     int width, int height, const double * intrinsics, double stamp,
                     const double * worldPose, int * nodeId) {
    auto & ctx = *static_cast<OnlineContext *>(context);
    try {
        cv::Mat color(height,width,CV_8UC3,const_cast<unsigned char *>(rgb));
        cv::Mat dep(height,width,CV_16UC1,const_cast<unsigned short *>(depth));
        rtabmap::CameraModel model("hp60c", intrinsics[0],intrinsics[1],intrinsics[2],intrinsics[3],
                                  rtabmap::CameraModel::opticalRotation(),0,cv::Size(width,height));
        rtabmap::SensorData data(color.clone(),dep.clone(),model,0,stamp);
        rtabmap::Transform opticalWorld(worldPose[0],worldPose[1],worldPose[2],worldPose[3],
                                        worldPose[4],worldPose[5],worldPose[6],worldPose[7],
                                        worldPose[8],worldPose[9],worldPose[10],worldPose[11]);
        auto robotWorld = opticalWorld * rtabmap::CameraModel::opticalRotation().inverse();
        if(!ctx.snapshotMode) {
            rtabmap::ParametersMap settings;
            settings["Mem/RehearsalSimilarity"] = "1.0";
            settings["RGBD/LinearUpdate"] = "0";
            settings["RGBD/AngularUpdate"] = "0";
            settings["Rtabmap/LoopThr"] = "1";
            settings["RGBD/ProximityBySpace"] = "false";
            settings["RGBD/ProximityByTime"] = "false";
            settings["RGBD/NeighborLinkRefining"] = "false";
            settings["Mem/BadSignaturesIgnored"] = "false";
            ctx.map.parseParameters(settings);
            ctx.snapshotMode = true;
        }
        auto started = std::chrono::steady_clock::now();
        if(!ctx.map.process(data,robotWorld,.0004f,.0004f)) return 0;
        ctx.timings[1] = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        int id = ctx.map.getLastLocationId();
        *nodeId = id;
        if(id > 0 && id != ctx.lastId) { ctx.lastId = id; return 2; }
        return id > 0 ? 1 : 0;
    } catch(const std::exception & e) { lastError=e.what(); return -1; }
}

// Authoritative geometry-only optimized graph, synchronized after each capture.
API int geometry_sync(void * context, const double * records, int count, const double * edges, int edgeCount) {
    auto & ctx = *static_cast<OnlineContext *>(context);
    try {
        auto optical = rtabmap::CameraModel::opticalRotation();
        auto readPose = [](const double * p) {
            return rtabmap::Transform(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11]);
        };
        std::map<int,rtabmap::Transform> poses;
        for(int i=0;i<count;++i) poses[int(records[i*13])] = readPose(records+i*13+1)*optical.inverse();
        ctx.geometryLinks.clear();
        cv::Mat information = cv::Mat::eye(6,6,CV_64FC1);
        for(int i=0;i<6;++i) information.at<double>(i,i) = i<3 ? 1.0/(.012*.012) : 1.0/(.02*.02);
        for(int i=0;i<edgeCount;++i) {
            const double * edge = edges+i*14;
            rtabmap::Link link(int(edge[0]),int(edge[1]),rtabmap::Link::kUserClosure,
                              optical*readPose(edge+2)*optical.inverse(),information);
            ctx.geometryLinks.emplace(link.from(),link);
        }
        ctx.map.setOptimizedPoses(poses,ctx.geometryLinks);
        ctx.geometryGraph = true;
        return 0;
    } catch(const std::exception & e) { lastError=e.what(); return -1; }
}
