#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp> //Thanks to Alessandro
#include <opencv2/opencv_modules.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/gpu.hpp>
#include <stdio.h>
#include "feature_matcher.h"

using namespace cv;

void cpuExtractor(Mat& img_1, Mat& img_2, std::vector<KeyPoint>& keypoints_1, std::vector<KeyPoint>& keypoints_2, 
					Mat& descriptors_1, Mat& descriptors_2, const string& feature_type) {
	//-- Step 1: Detect the keypoints using SURF Detector
	Ptr<FeatureDetector> detector_p = FeatureDetector::create(feature_type);
	Ptr<DescriptorExtractor> extractor_p = DescriptorExtractor::create(feature_type);
	
	detector_p->detect(img_1, keypoints_1);
	detector_p->detect(img_2, keypoints_2);
	//-- Step 2: Calculate descriptors (feature vectors)
	extractor_p->compute(img_1, keypoints_1, descriptors_1);
	extractor_p->compute(img_2, keypoints_2, descriptors_2);

	// delete detector_p;
	// delete extractor_p;
}

void gpuExtractor(Mat& img_1, Mat& img_2, std::vector<KeyPoint>& keypoints_1, std::vector<KeyPoint>& keypoints_2, 
					Mat& descriptors_1, Mat& descriptors_2) {
	// GPU SURF
	gpu::GpuMat img_g1, img_g2;  
 	gpu::GpuMat keypoints_g1, keypoints_g2;
 	gpu::GpuMat descriptors_g1, descriptors_g2;
	std::vector<float> descriptors_v1, descriptors_v2;

 	img_g1.upload(img_1);
 	img_g2.upload(img_2);

 	gpu::SURF_GPU FeatureFinder_gpu(400);
 	
 	FeatureFinder_gpu(img_g1, gpu::GpuMat(), keypoints_g1, descriptors_g1, false); 
 	FeatureFinder_gpu(img_g2, gpu::GpuMat(), keypoints_g2, descriptors_g2, false);
 	FeatureFinder_gpu.downloadKeypoints(keypoints_g1, keypoints_1);
 	FeatureFinder_gpu.downloadKeypoints(keypoints_g2, keypoints_2);
 	FeatureFinder_gpu.downloadDescriptors(descriptors_g1, descriptors_v1);  
 	FeatureFinder_gpu.downloadDescriptors(descriptors_g2, descriptors_v2);

 	descriptors_1 = cv::Mat(descriptors_v1);
 	descriptors_2 = cv::Mat(descriptors_v2);
}

/**
 * @function main
 * @brief Main function
 */
int matchFeatures(Mat& image1, Mat& image2, Mat& image_matches) {

	// feature type
	std::string feature_type = "ORB";


	// initialize data
	Mat img_1, img_2;
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	cvtColor(image1, img_1, CV_BGR2GRAY);
	cvtColor(image2, img_2, CV_BGR2GRAY);

	
	// choose detector type CPU or GPU
	// gpuExtractor(img_1, img_2, keypoints_1, keypoints_2, descriptors_1, descriptors_2);
	cpuExtractor(img_1, img_2, keypoints_1, keypoints_2, descriptors_1, descriptors_2, feature_type);


	// match descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	if (feature_type == "ORB") {
		// ORB is binary feature
		matcher = FlannBasedMatcher(new flann::LshIndexParams(20,10,2));
	}
	std::vector<DMatch> matches;
	// printf("before match\n");
	try{
		matcher.match(descriptors_1, descriptors_2, matches);
		
	} catch (...) {
		return 0;
	}
	// printf("after match\n");


	//Quick calculation of max and min distances between keypoints
	float min_dist, max_dist;
	for (int i = 0; i < descriptors_1.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;
		if (dist > max_dist)
			max_dist = dist;
	}


	// filter out good matches by the maximum distance determined statistically
	float max_distance;
	if (feature_type == "SIFT") {
		max_distance = 250;
	} else if (feature_type == "SURF") {
		max_distance = 0.2;
	} else if (feature_type == "ORB") {
		max_distance = 60;
	}

	
	// group matches by distance
	std::vector<DMatch> good_matches;
	std::vector<std::vector<DMatch> > good_matches_groups(4);
	for( int i = 0; i < descriptors_1.rows; i++ ) {
		if ( matches[i].distance < max_distance ) {
			good_matches.push_back( matches[i]);

			float score = matches[i].distance / max_distance;
			good_matches_groups[int(score/0.25)].push_back(matches[i]);

			// 0.0  -> (0, 0, 128)    (dark blue)
			// 0.25 -> (0, 255, 0)    (green)
			// 0.5  -> (255, 255, 0)  (yellow)
			// 0.75 -> (255, 128, 0)  (orange)
			// 1.0  -> (255, 0, 0)    (red)

		}
	}

	// print sample first 10 matches for debugging
	// printf("Good Matches: %d\n", good_matches.size());
	// printf("Groups Percentages: %.2f %.2f %.2f %.2f\n", 
	// 										good_matches_groups[0].size()*1.0/good_matches.size(),
	// 										good_matches_groups[1].size()*1.0/good_matches.size(),
	// 										good_matches_groups[2].size()*1.0/good_matches.size(),
	// 										good_matches_groups[3].size()*1.0/good_matches.size()
	// 										);
	// for (int i = 0; i < min(10, int(good_matches.size())); i++) {
	// 		printf("matches %d: dist: %g\n", i, good_matches[i].distance);
	// }


	// specify color grouped by distance
	vector<Scalar> colors(4);
	colors[0] = Scalar(128,0,0);
	colors[1] = Scalar(0,255,0);
	colors[2] = Scalar(0,255,255);
	colors[3] = Scalar(0,0,255);


	// draw good matches
	drawKeypoints(img_1, keypoints_1, img_1);
	drawKeypoints(img_2, keypoints_2, img_2);
	for (int i = 0, len = colors.size(); i < len; i++) {
		try {
			drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches_groups[i],
				image_matches, colors[i], Scalar::all(-1), vector<char>(),
				// mode: 0 -- default creates new output image
				// mode: 1 -- reuses output image
				i != 0);
		} catch (...) {
			continue;
		}
	}


	// if all bad qualities matches, ignore
	if (good_matches_groups[good_matches_groups.size()-1].size() * 1.0 / good_matches.size() > 0.7) {
		return 0;
	}

	// if (good_matches.size() < 30) {
	// 	return 0;
	// }

	// if

	return int(good_matches.size());
}