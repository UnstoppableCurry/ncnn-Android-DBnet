// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "Det.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cpu.h"

Det::Det()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}


int Det::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    dbNet.clear();
    dbNet.opt.lightmode = true;
//    dbNet.opt.num_threads = 4;
    dbNet.opt.use_packing_layout = true;




    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    dbNet.opt = ncnn::Option();
#if NCNN_VULKAN
    dbNet.opt.use_vulkan_compute = use_gpu;
#endif

    dbNet.opt.num_threads = ncnn::get_big_cpu_count();
    dbNet.opt.blob_allocator = &blob_pool_allocator;
    dbNet.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);


    dbNet.load_param(mgr, parampath);
    dbNet.load_model(mgr, modelpath);

//    dbNet.load_param(mgr, "pdocrv2.0_det-op.param");
//    dbNet.load_model(mgr, "pdocrv2.0_det-op.bin");
//    dbNet.load_param(mgr, "det-sim-op.param");
//    dbNet.load_model(mgr, "det-sim-op.bin");

    return 0;
}


std::vector<TextBox> Det::findRsBoxes(const cv::Mat& fMapMat, const cv::Mat& norfMapMat,
                                       const float boxScoreThresh, const float unClipRatio)
{
    float minArea = 3;
    std::vector<TextBox> rsBoxes;
    rsBoxes.clear();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); ++i)
    {
        float minSideLen, perimeter;
        std::vector<cv::Point> minBox = getMinBoxes(contours[i], minSideLen, perimeter);
        if (minSideLen < minArea)
            continue;
        float score = boxScoreFast(fMapMat, contours[i]);
        if (score < boxScoreThresh)
            continue;
        //---use clipper start---
        std::vector<cv::Point> clipBox = unClip(minBox, perimeter, unClipRatio);
        std::vector<cv::Point> clipMinBox = getMinBoxes(clipBox, minSideLen, perimeter);
        //---use clipper end---

        if (minSideLen < minArea + 2)
            continue;

        for (int j = 0; j < clipMinBox.size(); ++j)
        {
            clipMinBox[j].x = (clipMinBox[j].x / 1.0);
            clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), norfMapMat.cols);

            clipMinBox[j].y = (clipMinBox[j].y / 1.0);
            clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), norfMapMat.rows);
        }

        rsBoxes.emplace_back(TextBox{ clipMinBox, score });
    }
    reverse(rsBoxes.begin(), rsBoxes.end());

    return rsBoxes;
}
int Det::detect(const cv::Mat & src, float boxScoreThresh, float boxThresh, float unClipRatio)
{
    int width = src.cols;
    int height = src.rows;
    int target_size = 640;
    // pad to multiple of 32
    int w = width;
    int h = height;
    int input_w = width;
    int input_h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat input = ncnn::Mat::from_pixels_resize(src.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(input, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    const float meanValues[3] = { 0.485 * 255, 0.456 * 255, 0.406 * 255 };
    const float normValues[3] = { 1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0, 1.0 / 0.225 / 255.0 };

    in_pad.substract_mean_normalize(meanValues, normValues);
    ncnn::Extractor extractor = dbNet.create_extractor();

    extractor.input("input0", in_pad);
    ncnn::Mat out;
    extractor.extract("out1", out);

    cv::Mat fMapMat(in_pad.h, in_pad.w, CV_32FC1, (float*)out.data);
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;

    cv::dilate(norfMapMat, norfMapMat, cv::Mat(), cv::Point(-1, -1), 1);

    cv::Mat norfMapCrop;
    cv::Rect rect(wpad/2, hpad/2, input_w, input_h);
    norfMapMat(rect).copyTo(norfMapCrop);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(norfMapCrop, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    cv::drawContours(src, contours, -1, cv::Scalar(255,255,255), 1, 1);


    std::vector<TextBox> result = findRsBoxes(fMapMat, norfMapMat, boxScoreThresh, 2.0f);

    for(int i = 0; i < result.size(); i++)
    {
        std::vector<cv::Point> points;

        for(int j = 0; j < result[i].boxPoint.size(); j++)
        {
            float x = (result[i].boxPoint[j].x-(wpad/2))/scale;
            float y = (result[i].boxPoint[j].y-(hpad/2))/scale;
            x = std::max(std::min(x,(float)(width-1)),0.f);
            y = std::max(std::min(y,(float)(height-1)),0.f);
            result[i].boxPoint[j].x = x;
            result[i].boxPoint[j].y = y;
            points.push_back(cv::Point(x, y));

        }
        if (points.size() >= 4) {
            cv::polylines(src, points, true, cv::Scalar(255, 255, 255), 1, 8, 0);
        }
    }

    return 0;
}

//int Det::draw(cv::Mat& image, const std::vector<TextBox>& textBoxes)
//{
//
//    for (const auto& textBox : textBoxes) {
//        std::vector<cv::Point> points;
//        // 将你的boxPoint转换为cv::Point
//        for (const auto& point : textBox.boxPoint) {
//            points.push_back(cv::Point(point.x, point.y));
//        }
//        // 假设你的boxPoint形成一个封闭的形状（例如，四边形）
//        if (points.size() >= 4) {
//            cv::polylines(image, points, true, cv::Scalar(0, 0, 255), 2, 8, 0);
//        }
//    }
//
//    return 0;
//}


