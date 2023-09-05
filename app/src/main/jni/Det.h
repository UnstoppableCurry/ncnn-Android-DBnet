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

#ifndef DET_H
#define DET_H

#include <opencv2/core/core.hpp>
#include <net.h>
#include "common.h"

class Det
{
public:
    Det();

    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int detect(const cv::Mat & src, float boxScoreThresh, float boxThresh, float unClipRatio);
    std::vector<TextBox> findRsBoxes(const cv::Mat& fMapMat, const cv::Mat& norfMapMat, const float boxScoreThresh, const float unClipRatio);
//    int draw(cv::Mat& image, const std::vector<TextBox>& textBoxes);

private:


    static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    static ncnn::PoolAllocator g_workspace_pool_allocator;
    const int dstHeight = 32;//when use PP-OCRv3 it should be 48
    ncnn::Net dbNet;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif
