
/*
The MIT License
Copyright (c) 2015 Satyaki Chakraborty

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <sstream>
#include "iostream"
#include "vector"

#include "DualEstereoVisualOdometry.h"

#define PI 3.14159265
#define minFeatures 35

using namespace std;
using namespace cv;

string to_string(int i){
	string result;          // string which will contain the result
	ostringstream convert;   // stream used for the conversion
	convert << i;      // insert the textual representation of 'Number' in the characters in the stream
	result = convert.str();
	return result;
}
string get_sequence(int n) {
	if (n==0) return "000000";
	else if (n/10 == 0) return "00000"+to_string(n);
	else if (n/100 == 0) return "0000"+to_string(n);
	else if (n/1000 == 0) return "000"+to_string(n);
	else if (n/10000 == 0) return "00"+to_string(n);
	return NULL;
}
Mat stackH(Mat im1,Mat im2){
    Size sz1 = im1.size();
    Size sz2 = im2.size();
    Mat im3(sz1.height, sz1.width+sz2.width, CV_8UC3);
    Mat left(im3, Rect(0, 0, sz1.width, sz1.height));
    im1.copyTo(left);
    Mat right(im3, Rect(sz1.width, 0, sz2.width, sz2.height));
    im2.copyTo(right);
    //imshow("im3", im3);
    return im3;
}
Mat stackV(Mat im1,Mat im2){
    Size sz1 = im1.size();
    Size sz2 = im2.size();
    Mat im3(sz1.height+sz2.height, sz1.width, CV_8UC3);
    Mat top(im3, Rect(0, 0, sz1.width, sz1.height));
    im1.copyTo(top);
    Mat down(im3, Rect(0, sz1.height, sz2.width, sz2.height));
    im2.copyTo(down);
    //imshow("im3", im3);
    return im3;
}
int main(int argc, char** argv) {

	if (argc< 3) {
		cout << "Enter path to data.. ./odo <path> <numFiles>\n";
		return -1;
	}
	
	if (argv[1][strlen(argv[1])-1] == '/') {
		argv[1][strlen(argv[1])-1] = '\0';
	}
	string path = string(argv[1]);
	int SEQ_MAX = atoi(argv[2]);

	//int seq_id = 0, scene_id = 0;
	Mat top_view2 = Mat::zeros(400, 400, CV_8UC3);
	Mat top_view3 = Mat::zeros(400, 400, CV_8UC3);
	Mat cur_frame_c2,cur_frame_c3;
	Mat t2_,t3_;
	cur_frame_c2 = imread(path+"/image_2/000000.png");
	cur_frame_c3= imread(path+"/image_3/000000.png");
	DualEstereoVisualOdometry dsvo2(cur_frame_c2,cur_frame_c3);
	/* some function tests
	vector<Point3f> prev3Dpts;
	vector<Point2f> cur2DforPnP;
	prev3Dpts.push_back(Point3f( 1, 0, 10));
	prev3Dpts.push_back(Point3f( 1, 0, 20));
	prev3Dpts.push_back(Point3f(-1, 0, 20));
	prev3Dpts.push_back(Point3f(-1, 0, 10));
	prev3Dpts.push_back(Point3f( 0, 0, 10));
	prev3Dpts.push_back(Point3f(-1, 0, 10));
	prev3Dpts.push_back(Point3f(-1,-1, 10));
	prev3Dpts.push_back(Point3f( 1,-1, 10));
	prev3Dpts.push_back(Point3f( 1, 0, 10));
	prev3Dpts.push_back(Point3f( 1,-1, 10));
	prev3Dpts.push_back(Point3f( 1, 1, 10));
	prev3Dpts.push_back(Point3f( 0, 0, 15));
	prev3Dpts.push_back(Point3f(-1, 0, 15));
	prev3Dpts.push_back(Point3f(-1,-1, 15));
	prev3Dpts.push_back(Point3f( 1,-1, 15));
	prev3Dpts.push_back(Point3f( 1, 0, 15));
	prev3Dpts.push_back(Point3f( 1,-1, 15));
	prev3Dpts.push_back(Point3f( 1, 1, 15));
	Mat rvec,tll,inliers;
	tll = (Mat_<double>(3, 1) << 0., 1., 0.);
	rvec = (Mat_<double>(3, 1) << PI/16.0, 0*PI/8.0, 0.);
	cout << "rvec" << rvec << endl;
	cout << "tll=" << tll << endl;
	Mat K=dsvo2.K;
	cv::projectPoints(prev3Dpts,rvec,tll,K,Mat(),cur2DforPnP);
	//Transformation estimation 3D-3D
	vector<Point3f> cur3Dpts;
	Mat R,t;
	Rodrigues(rvec,R);
	Mat T=(Mat_<double>(3, 4) << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), tll.at<double>(0,0),
			                     R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), tll.at<double>(1,0),
								 R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), tll.at<double>(2,0));
	cout << "R"<< R <<endl;
	cout << "t"<< t <<endl;
	cv::transform(prev3Dpts,cur3Dpts,T);
	Mat  cur3DMat=Mat( cur3Dpts).reshape(1);
	Mat prev3DMat=Mat(prev3Dpts).reshape(1);
	//cout << "centerR(prev3Dpts)"<<centerR(prev3DMat)<<endl;
	//cout << "centerR(prev3Dpts)"<<prev3DMat<<endl;
	rigidTransformation(prev3DMat,cur3DMat,R,t);
	cout << "R"<< R <<endl;
	cout << "t"<< t <<endl;


	Mat pimg(380,1200, CV_8UC3, Scalar(255,255,255));
	line(pimg,cur2DforPnP[0],cur2DforPnP[1],Scalar(255,0,0));
	line(pimg,cur2DforPnP[1],cur2DforPnP[2],Scalar(255,0,0));
	line(pimg,cur2DforPnP[2],cur2DforPnP[3],Scalar(255,0,0));
	line(pimg,cur2DforPnP[3],cur2DforPnP[0],Scalar(255,0,0));
	for(unsigned int i=4;i<cur2DforPnP.size();i++){
		if(i>11)
			circle(pimg,cur2DforPnP[i],2,Scalar(0,0,255));
		else
			circle(pimg,cur2DforPnP[i],2,Scalar(0,255,0));
	}
	imshow("pimg",pimg);
	//Transformation estimation 3D-2D
	solvePnPRansac(prev3Dpts, cur2DforPnP, K, Mat(), rvec, tll,false,500,2,0.99,inliers, SOLVEPNP_ITERATIVE );//SOLVEPNP_EPNP

	Rodrigues(rvec,R);
	t=-R.t()*tll;
	cout << "rvec" << rvec << endl;
	cout << "tll=" << tll << endl;
	cout << "t=" << t << endl;
	cout << inliers.size() << endl;
	waitKey(0);
    */
	float l=0;//trajectory length
	for (int i=1; i<=SEQ_MAX; i+=1) {
		string nf2=path+"/image_2/"+get_sequence(i)+".png";
		string nf3=path+"/image_3/"+get_sequence(i)+".png";
		cout << nf2 << endl;
		cur_frame_c2 = imread(nf2);
		cur_frame_c3 = imread(nf3);
		dsvo2.stepStereoOdometry(cur_frame_c2,cur_frame_c3);
		t2_=dsvo2.tgl;
		cout << "gpos:" << t2_.t() << endl;
		l+=abs(dsvo2.tll.at<float>(2,0));
		cout << "lpos:" << dsvo2.tll.t() <<":"<<l<< endl;
		circle(top_view2, Point(200+t2_.at<double>(0, 2)/2.0, (200+t2_.at<double>(0, 0)/2.0)), 3, Scalar(0, 255, 0), -1);
		circle(top_view2, Point(200+t2_.at<double>(0, 2)/2.0, (200+t2_.at<double>(0, 0)/2.0)), 2, Scalar(0, 0, 255), -1);
  		imshow("Top view2", top_view2);
 		if (waitKey(0) == 27) break;
	}
	return 0;
}

