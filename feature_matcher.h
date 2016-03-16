/*
 * FeatureMatcher.hpp
 *
 *  Created on: Mar 13, 2016
 *      Author: daranday
 */

#ifndef SRC_FEATURE_MATCHER_H_
#define SRC_FEATURE_MATCHER_H_

#include <opencv2/core/core.hpp>
// #include <utility>

int matchFeatures(cv::Mat& image1, cv::Mat& image2, cv::Mat& image_matches);


#endif /* SRC_FEATURE_MATCHER_H_ */
