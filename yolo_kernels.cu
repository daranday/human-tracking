#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include <sys/time.h>
}

#ifdef OPENCV
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
extern "C" image ipl_to_image(IplImage* src);
extern "C" void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);
extern "C" void draw_yolo(image im, int num, float thresh, box *boxes, float **probs);

extern "C" char *voc_names[];
extern "C" image voc_labels[];

static float **probs;
static box *boxes;
static network net;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp ;
static cv::VideoCapture cap;
static float fps = 0;
static float demo_thresh = 0;

// new stuff
#include "feature_matcher.h"
#include <vector>
#include <string>
#include <deque>
std::vector<std::deque<cv::Mat> > person_db;
int pool_size = 2;
int wait_period = 1;
int since_last = 0;
std::vector<int> active;
cv::Mat current_img;
std::vector<cv::Mat> image_matches;
std::vector<cv::Mat> bad_matches;
std::vector<int> indices_matches;
bool no_match;
IplImage* im_ptr = NULL;
int frame_num = 0;

void *fetch_in_thread(void *ptr)
{
    cv::Mat frame_m;
    cap >> frame_m;
    IplImage frame = frame_m;
    in = ipl_to_image(&frame);
    rgbgr_image(in);
    in_s = resize_image(in, net.w, net.h);
    ++frame_num;
    return 0;
}

void image_to_mat(image p, cv::Mat& m) {
    int x,y,k;
    image copy = copy_image(p);
    constrain_image(copy);
    if(p.c == 3) rgbgr_image(copy);
    //normalize_image(copy);

    // char buff[256];
    // //sprintf(buff, "%s (%d)", name, windows);
    // sprintf(buff, "%s", name);

    m.create(p.h, p.w, CV_8UC3);

    // IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    // int step = disp->widthStep;
    // cvNamedWindow(buff, CV_WINDOW_NORMAL); 
    //cvMoveWindow(buff, 100*(windows%10) + 200*(windows/10), 100*(windows%10));
    // ++windows;
    for(y = 0; y < p.h; ++y){
        for(x = 0; x < p.w; ++x){
            for(k= 0; k < p.c; ++k){
                m.at<cv::Vec3b>(y,x)[k] = (unsigned char)(get_pixel(copy,x,y,k)*255);
                // m.at<uchar>(y, x, 0) = 255; //(unsigned char)(get_pixel(copy,x,y,k)*255);
                // m.at<uchar>(y, x, 1) = 0; //(unsigned char)(get_pixel(copy,x,y,k)*255);
                // m.at<uchar>(y, x, 2) = 0; //(unsigned char)(get_pixel(copy,x,y,k)*255);
                // disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy,x,y,k)*255);
            }
        }
    }
    free_image(copy);

    // m = cv::Mat(disp);
    // return disp;
}

void track_person(image image_im, int num, float thresh, box *boxes, float **probs, char **names, image *labels, int classes)
{
    int cls_person = 14;

    active = std::vector<int>(person_db.size());
    std::vector<cv::Rect> rects;
    std::vector<int> person_ids;
    image_matches.clear();
    indices_matches.clear();
    bad_matches.clear();

    image_to_mat(image_im, current_img);

    for(int i = 0; i < num; ++i){
        int cls = max_index(probs[i], classes);
        float prob = probs[i][cls];
        if(cls == cls_person && prob > thresh){

            box& b = boxes[i];
            int left  = (b.x-b.w/2.)*image_im.w;
            int right = (b.x+b.w/2.)*image_im.w;
            int top   = (b.y-b.h/2.)*image_im.h;
            int bot   = (b.y+b.h/2.)*image_im.h;

            if(left < 0) left = 0;
            if(right > image_im.w-1) right = image_im.w-1;
            if(top < 0) top = 0;
            if(bot > image_im.h-1) bot = image_im.h-1;


            cv::Rect rect(left, top, right-left, bot-top);
            cv::Mat new_box = current_img(rect);

            int found = -1;
            int max_person = -1;
            int max_matches = 0;
            cv::Mat max_image_match;

            // search match between current person with person database
            for (int j = 0, len = person_db.size(); j < len; ++j) {
                if (active[j] == 0) {
                    int vote = 0;
                    cv::Mat image_match;
                    for (int k = 0, len = person_db[j].size(); k < len; ++k) {
                        int match_result = matchFeatures(person_db[j][k], new_box, image_match);

                        if (match_result > 0) {
                            vote++;
                        }  else {
                            bad_matches.push_back(image_match);
                        }
                    }
                    if (vote >= person_db[j].size()/2) {
                        max_person = j;
                        max_matches = vote;
                        max_image_match = image_match;
                        break;
                    }
                }
            }

            // found person, update old person portfolio 
            if (max_person != -1) {
                found = max_person;
                if (since_last < wait_period) {
                    ++since_last;
                } else {
                    person_db[max_person].push_back(new_box.clone());
                    since_last = 0;
                }
                active[max_person] = max_matches;
                image_matches.push_back(max_image_match);
                indices_matches.push_back(max_person+1);
            }

            // did not find any person, creating a new profile in person database
            if (found == -1) {
                if (person_db.size() == 0) {
                    found = person_db.size();
                    person_db.push_back(std::deque<cv::Mat>());
                    person_db.back().push_back(new_box.clone());
                    active.push_back(1);
                }
            }

            if (found != -1) {
                rects.push_back(rect);
                person_ids.push_back(found+1);
                if (person_db[found].size() > pool_size) {
                    person_db[found].pop_front();
                }
            } else {
                rects.push_back(rect);
                person_ids.push_back(0);
            }
        }
    }

    for (int i = 0, len = person_ids.size(); i < len; ++i) {
        // label person
        char person_callname[50];
        sprintf(person_callname, "Person %d", person_ids[i]);
        if (person_ids[i])
            printf("Person %d, matches %d\n", person_ids[i], active[person_ids[i]-1]);
        cv::putText(current_img, person_callname, cv::Point(rects[i].x+10, rects[i].y+30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar( 255,0,0 ), 2);
        cv::rectangle(current_img, rects[i], cv::Scalar(240,128,128), 3);
    }

    if (image_matches.size()) {
        printf("Match accepted!\n");
        no_match = false;
    } else {
        printf("Match rejected or no match!\n");
        no_match = true;
    }
}

void *detect_in_thread(void *ptr)
{
    float nms = .4;

    detection_layer l = net.layers[net.n-1];
    float *X = det_s.data;
    float *predictions = network_predict(net, X);
    free_image(det_s);
    convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxes, 0);
    if (nms > 0) do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
    // printf("\033[2J");
    // printf("\033[1;1H");
    printf("\nFPS:%.0f\n",fps);
    printf("Objects:\n\n");

    // new stuff
    track_person(det, l.side*l.side*l.n, demo_thresh, boxes, probs, voc_names, voc_labels, 20);
    // draw_detections(det, l.side*l.side*l.n, demo_thresh, boxes, probs, voc_names, voc_labels, 20);

    return 0;
}

extern "C" void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index)
{
    demo_thresh = thresh;
    printf("YOLO demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);

    srand(2222222);


    bool use_video = false;
    if (use_video) {
        // Open video file
        std::string video_path = "drone.mp4";
        cv::VideoCapture vid(video_path);
        cap = vid;
        if(!cap.isOpened()) error(("Couldn't open video: " + video_path + "\n").c_str());
    } else {
        // Open camera
        cv::VideoCapture cam(cam_index);
        cap = cam;
        if(!cap.isOpened()) error("Couldn't connect to webcam.\n");
    }

    detection_layer l = net.layers[net.n-1];
    int j;

    boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
    probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

    pthread_t fetch_thread;
    pthread_t detect_thread;

    fetch_in_thread(0);
    det = in;
    det_s = in_s;

    fetch_in_thread(0);
    detect_in_thread(0);
    disp = det;
    det = in;
    det_s = in_s;

    int fast_forward = 1;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);
        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);

        // if (person_db.size()) {
        //     for (int i = 0, len = person_db[0].size(); i < len; ++i) {
        //         if (person_db[0][i].rows) {
        //             char match_name[50];
        //             sprintf(match_name, "Person 1 Sample %d", i);
        //             cv::imshow(match_name, person_db[0][i]);
        //             char key = cv::waitKey(1);
        //             if (key == 's') {
        //                 fast_forward = 0;
        //             }
        //         }
        //     }
            
        // }
        for (int i = 0, len = image_matches.size(); i < len; ++i) {
            if (image_matches[i].rows) {
                char match_name[50];
                sprintf(match_name, "Match %d", indices_matches[i]);
                cv::imshow(match_name, image_matches[i]);
                char key = cv::waitKey(1);
                if (key == 's') {
                    fast_forward = 0;
                }
            }
        }
        // for (int i = 0, len = bad_matches.size(); i < len; ++i) {
        //     if (bad_matches[i].rows) {
        //         char match_name[50];
        //         sprintf(match_name, "Bad Match %d", i);
        //         cv::imshow(match_name, bad_matches[i]);
        //         char key = cv::waitKey(1);
        //         if (key == 's') {
        //             fast_forward = 0;
        //         }
        //     }
        // }
        if (current_img.rows) {
            cv::imshow("YOLO", current_img);
            char key = cv::waitKey(1);
            if (key == 's') {
                fast_forward = 0;
            }
        }

        printf("Frame: %d\n", frame_num);

        if (fast_forward == 0) {
            char key = cv::waitKey(0);
            if (key == 'f') {
                fast_forward = 1;
            } else if (key == 's') {
                fast_forward = 0;
            }
        }
        

        // show_image(disp, "YOLO");
        free_image(disp);
        cvWaitKey(1);

        disp  = det;
        det   = in;
        det_s = in_s;

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
}
#else
extern "C" void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index){
    fprintf(stderr, "YOLO demo needs OpenCV for webcam images.\n");
}
#endif

