/*
 * MonoVisualOdometry.h
 *
 *  Created on: Jun 19, 2016
 *      Author: francisco
 */

#ifndef DUALESTEREOVISUALODOMETRY_H_
#define DUALESTEREOVISUALODOMETRY_H_
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;

string type2str(int type) {
 string r;

 uchar depth = type & CV_MAT_DEPTH_MASK;
 uchar chans = 1 + (type >> CV_CN_SHIFT);

 switch ( depth ) {
   case CV_8U:  r = "8U"; break;
   case CV_8S:  r = "8S"; break;
   case CV_16U: r = "16U"; break;
   case CV_16S: r = "16S"; break;
   case CV_32S: r = "32S"; break;
   case CV_32F: r = "32F"; break;
   case CV_64F: r = "64F"; break;
   default:     r = "User"; break;
 }

 r += "C";
 r += (chans+'0');

 return r;
}

Mat centroid(Mat &m){
	Mat t=m.row(0);
	int n=m.rows;
	for(int i=1;i<n;i++){
		t+=m.row(i);
	}
	t/=n;
	return t;
}
Mat centroidC(Mat &m){
	Mat t=m.col(0);
	int n=m.cols;
	for(int i=1;i<n;i++){
		t+=m.col(i);
	}
	t/=n;
	return t;
}
Mat centerR(Mat &M){
	Mat Ret;
	M.copyTo(Ret);
	Mat centroidM(centroid(M));
	int n=M.rows;
	for(int i=0;i<n;i++){
		Ret.row(i)-=centroidM;
	}
	return Ret;
}
Mat centerC(Mat &M){
	Mat Ret;
	M.copyTo(Ret);
	Mat centroidM(centroidC(M));
	int n=M.cols;
	for(int i=0;i<n;i++){
		Ret.col(i)-=centroidM;
	}
	return Ret;
}

void rigidTransformation(Mat &A,Mat &B,Mat &R,Mat &t){
	Mat centroidA,centroidB;
	centroidA=centroid(A);
	centroidB=centroid(B);
	Mat AA=centerR(A);
	//cout << A.rows<<A.cols<<A.channels() << "AA"<<A<<endl;
	Mat BB=centerR(B);
	Mat H=AA.t()*BB;
	Mat w, u, vt;
	SVD::compute(H, w, u, vt);
	R=vt.t()*u.t();
	if(cv::determinant(R)<0.0){
		vt.row(2)*=-1;
		R=vt.t()*u.t();
	}
	t=-R*centroidA.t()+centroidB.t();

	/*http://nghiaho.com/?page_id=671
	N = A.shape[0]; # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)

    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = transpose(AA) * BB

    U, S, Vt = linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
       print "Reflection detected"
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T
	 */
}

class DualEstereoVisualOdometry {
public :
	string to_string(float i){
		string result;          // string which will contain the result
		ostringstream convert;   // stream used for the conversion
		convert.precision(2);
		convert << i;      // insert the textual representation of 'Number' in the characters in the stream
		result = convert.str();
		return result;
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
	Mat curFrameL,curFrameL_c, prevFrameL,prevFrameL_c, curFrameL_kp, prevFrameL_kp;
	Mat curFrameR,curFrameR_c, prevFrameR,prevFrameR_c, curFrameR_kp, prevFrameR_kp;
	vector<KeyPoint> curKeypointsL, prevKeypointsL, curGoodKeypointsL, prevGoodKeypointsL;
	vector<KeyPoint> curKeypointsR, prevKeypointsR, curGoodKeypointsR, prevGoodKeypointsR;
	Mat curDescriptorsL, prevDescriptorsL;
	Mat curDescriptorsR, prevDescriptorsR;
	vector<Point2f> curPointsL,prevPointsL;
	vector<Point2f> curPointsR,prevPointsR;
	vector<DMatch> goodMatchesL;
	vector<DMatch> goodMatchesR;

	Mat descriptors_1, descriptors_2,  img_matches;
	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
	Ptr<flann::IndexParams> indexParams;
	Ptr<flann::SearchParams> searchParams;
	Ptr<DescriptorMatcher> matcher;
	// relative scale
	double scale;
	//double f = (double)(8.941981e+02 + 8.927151e+02)/2;
	//Point2f pp((float)6.601406e+02, (float)2.611004e+02);
	double f ; // focal length in pixels as in K intrinsic matrix
	Point2f pp; //principal point in pixel
	Mat K; //intrinsic matrix
	//global rotation and translation
	Mat Rgl, tgl,Rglprev,tglprev;
	Mat Rgr, tgr,Rgrprev,tgrprev;
	Mat Rg, tg,Rgprev,tgprev;
	//local rotation and transalation from prev to cur
	Mat Rll, tll;
	Mat Rlr, tlr;
	Mat Rl, tr;
	//STEREO DATA
	Mat curDisp,prevDisp;
	Mat curPointCloud,prevPointCloud;
	//Stereo Matches
	Ptr<StereoMatcher> sm;
	//Reprojection Matrix from 2D (u,v) and disp to 3D (X,Y,Z)
	Mat Q;
	double baseLine;
	//temp attributes
	vector<vector<DMatch> > matches;
	Mat E,mask;
	vector<Point3f> prev3Dpts,cur3Dpts;
//public:
	DualEstereoVisualOdometry(Mat &pcurFrameL_c,Mat &pcurFrameR_c){
		pcurFrameL_c.copyTo(curFrameL_c);
		pcurFrameR_c.copyTo(curFrameR_c);
		cvtColor(curFrameL_c, curFrameL, CV_BGR2GRAY);
		cvtColor(curFrameR_c, curFrameR, CV_BGR2GRAY);

		sm=StereoBM::create(16*4,11);
		sm->compute(curFrameL,curFrameR,curDisp);
		detector = ORB::create(1000);
		extractor = ORB::create();
		indexParams = makePtr<flann::LshIndexParams> (6, 12, 1);
		searchParams = makePtr<flann::SearchParams>(50);
		matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);

		//detector->detect(curFrameL, curKeypointsL);
		binnedDetection(curFrameL, curKeypointsL);
		//detector->detect(curFrameR, curKeypointsR);
		binnedDetection(curFrameR, curKeypointsR);
		extractor->compute(curFrameL, curKeypointsL, curDescriptorsL);
		extractor->compute(curFrameR, curKeypointsR, curDescriptorsR);
        //Global transformations
		Rg  = (Mat_<double>(3, 3) << 1., 0., 0., 0., 1., 0., 0., 0., 1.);
		tg  = (Mat_<double>(3, 1) << 0., 0., 0.);
		Rgl = (Mat_<double>(3, 3) << 1., 0., 0., 0., 1., 0., 0., 0., 1.);
		tgl = (Mat_<double>(3, 1) << 0., 0., 0.);
		Rgr = (Mat_<double>(3, 3) << 1., 0., 0., 0., 1., 0., 0., 0., 1.);
		tgr = (Mat_<double>(3, 1) << 0., 0., 0.);
		//Local transformations
		Rll = (Mat_<double>(3, 3) << 1., 0., 0., 0., 1., 0., 0., 0., 1.);
		tll = (Mat_<double>(3, 1) << 0., 0., 0.);
		Rlr = (Mat_<double>(3, 3) << 1., 0., 0., 0., 1., 0., 0., 0., 1.);
		tlr = (Mat_<double>(3, 1) << 0., 0., 0.);

        // relative scale
        scale = 1.0;

		//f = (double)(8.941981e+02 + 8.927151e+02)/2;
		//pp((float)6.601406e+02, (float)2.611004e+02);
        //Intrinsic kitti values 00
		f = (double)(7.188560000000e+02 + 7.188560000000e+02)/2;
		pp=Point2f((float)6.071928000000e+02, (float)1.852157000000e+02);
		baseLine=3.861448000000e+02/f*16;//base = -P2_roi(1,4)/P2_roi(1,1)
	    double cx=pp.x;
	    double cy=pp.y;
	    double Tx=baseLine;
	    //Intrinsic Matrix
		K = (Mat_<double>(3, 3) << f   ,  0.00, cx,
				                   0.00,  f   , cy,
								   0.00,  0.00, 1.00);
		//reprojection Matrix
		Q = (Mat_<double>(4, 4) << 1.00,  0.00, 0.00, -cx,
				                   0.00,  1.00, 0.00, -cy,  // turn points 180 deg around x-axis,
								   0.00,  0.00, 0.00,  f,     // so that y-axis looks up
								   0.00,  0.00, -1./Tx,  0);
        reprojectImageTo3D(curDisp, curPointCloud, Q, true);
	}
	virtual ~DualEstereoVisualOdometry(){}
	// function performs ratiotest
	// to determine the best keypoint matches
	// between consecutive poses
	void ratioTest(vector<vector<DMatch> > &matches, vector<DMatch> &good_matches) {
		for (vector<vector<DMatch> >::iterator it = matches.begin(); it!=matches.end(); it++) {
			if (it->size()>1 ) {
				if ((*it)[0].distance/(*it)[1].distance > 0.6f) {
					it->clear();
				}
			} else {
				it->clear();
			}
			if (!it->empty()) good_matches.push_back((*it)[0]);
		}
	}
	inline void findGoodMatches(vector<KeyPoint> &keypoints_1,Mat &descriptors_1,
			                    vector<KeyPoint> &keypoints_2,Mat &descriptors_2,
						        vector<DMatch>   &good_matches,
						        vector<KeyPoint> &good_keypoints_1, vector<KeyPoint> &good_keypoints_2){
		matches.clear();
		good_matches.clear();

		try {
			matcher->knnMatch(descriptors_1, descriptors_2, matches, 2);
			ratioTest(matches, good_matches);
		} catch(Exception &e) {
			//cerr << "knnMatch error"<<endl;;
		}

		good_keypoints_1.clear();
		good_keypoints_2.clear();
		for ( size_t m = 0; m < good_matches.size(); m++) {
			int i1 = good_matches[m].queryIdx;
			int i2 = good_matches[m].trainIdx;
			CV_Assert(i1 >= 0 && i1 < static_cast<int>(keypoints_1.size()));
            CV_Assert(i2 >= 0 && i2 < static_cast<int>(keypoints_2.size()));
            good_keypoints_1.push_back(keypoints_1[i1]);
            good_keypoints_2.push_back(keypoints_2[i2]);
		}
	}
	inline void getPoseFromEssentialMat(vector<Point2f> &point1, vector<Point2f> &point2,Mat &R,Mat &t){
		E = findEssentialMat(point2, point1, f, pp, RANSAC, 0.999, 1.0,mask);
		recoverPose(E, point2, point1, R, t, f, pp,mask);
	}
	void getUVdispFromKeyPoints(vector<KeyPoint> &kpts,Mat &disp,Mat &uvd){
		vector<Point3f> vp;
		Point3f p;
		for(unsigned int i=0;i<kpts.size();i++){
			KeyPoint &kp=kpts[i];
			float u=kp.pt.x;
			float v=kp.pt.y;
		    float d=disp.at<float>(u,v);
			p.x=u;
			p.y=v;
			p.z=d;
			vp.push_back(p);
		}
		uvd= Mat(vp);
	}
	void binnedDetection(Mat &img,vector<KeyPoint> &kpts){
		int bwc=10;//bin with col
		int bhr=5;//bin height row
		int brows=img.rows/bhr;
		int bcols=img.cols/bwc;
		kpts.clear();
		vector<KeyPoint> dkpt;
		Mat ims;
		for(int i=0;i<img.rows-brows;i+=brows/2)
			for(int j=0;j<img.cols-bcols;j+=bcols/2){
				Rect r(j, i, bcols, brows);
				Mat imgBin(img, r);
				detector->detect(imgBin, dkpt);
				//relocate
				for(unsigned int k=0;k<dkpt.size();k++){
					dkpt[k].pt+=Point2f(j,i);
				}
				//cv::drawKeypoints(img,dkpt,ims, Scalar(0,0,255));
				//imshow("imgBin",imgBin);
				//imshow("dkpt",ims);
				//waitKey(0);
				//cout << "#dkpt="<<dkpt.size()<<endl;
			    kpts.insert(kpts.end(), dkpt.begin(), dkpt.end());//append dkpt to kpts
				/*int n=5;
				if(dkpt.size()>5){
					for(int i=0;i<n;i++){
					kpts.push_back(dkpt.back());
					dkpt.pop_back();
					}
				}
				else{
				    kpts.insert(kpts.end(), dkpt.begin(), dkpt.end());//append dkpt to kpts
				}*/
			}
		//cv::drawKeypoints(img,kpts,ims, Scalar(0,0,255));
		//imshow("kpts",ims);
		//waitKey(0);
	}
	void stepStereoOdometry(Mat& pcurFrameL_c,Mat& pcurFrameR_c){
		curFrameL.copyTo(prevFrameL);		curFrameR.copyTo(prevFrameR);
		curFrameL_c.copyTo(prevFrameL_c);	curFrameR_c.copyTo(prevFrameR_c);
		curDisp.copyTo(prevDisp);
		curPointCloud.copyTo(prevPointCloud);
		pcurFrameL_c.copyTo(curFrameL_c);	pcurFrameR_c.copyTo(curFrameR_c);
		cvtColor(curFrameL_c, curFrameL, CV_BGR2GRAY);		cvtColor(curFrameR_c, curFrameR, CV_BGR2GRAY);
		prevKeypointsL = curKeypointsL;		prevKeypointsR = curKeypointsR;
		curDescriptorsL.copyTo(prevDescriptorsL);		curDescriptorsR.copyTo(prevDescriptorsR);
		prevPointsL = curPointsL;    		prevPointsR = curPointsR;

		binnedDetection(curFrameL, curKeypointsL);
		//detector->detect(curFrameR, curKeypointsR);
		extractor->compute(curFrameL, curKeypointsL, curDescriptorsL);
		//extractor->compute(curFrameR, curKeypointsR, curDescriptorsR);

		//prev to cur left matches
		findGoodMatches(prevKeypointsL,prevDescriptorsL,
				        curKeypointsL, curDescriptorsL,
						goodMatchesL,
						prevGoodKeypointsL,curGoodKeypointsL);
		cout << "#prevKeypointsL="<<prevKeypointsL.size()<<endl;
		cout << "#curKeypointsL="<<curKeypointsL.size()<<endl;
		cout << "#prevGoodKeypointsL="<<prevGoodKeypointsL.size()<<endl;
		cout << "#curGoodKeypointsL="<<curGoodKeypointsL.size()<<endl;

		sm->compute(curFrameL, curFrameR, curDisp);
        cout <<"curDisp.type"<<type2str(curDisp.type())<< endl;
        reprojectImageTo3D(curDisp, curPointCloud, Q, true);

        Mat disp8;
		Mat dispC8;
        prevDisp.convertTo(disp8, CV_8U);
		cvtColor(disp8,dispC8,CV_GRAY2BGR);
		addWeighted(dispC8,0.2,prevFrameL_c,0.8,0.0,dispC8);

        Mat cdisp8;
		Mat cdispC8;
		curDisp.convertTo(cdisp8,CV_8U);
		cvtColor(cdisp8,cdispC8,CV_GRAY2BGR);
		addWeighted(cdispC8,0.2,curFrameL_c,0.8,0.0,cdispC8);
		Mat cpImg=stackV(cdispC8,dispC8);

		curPointsL.clear();
		KeyPoint::convert(curGoodKeypointsL, curPointsL);
		prevPointsL.clear();
		KeyPoint::convert(prevGoodKeypointsL, prevPointsL);
		//Better points with outlayer detection with essential ransac
//		Mat R,t;
//		if (prevPointsL.size() >5 && curPointsL.size() > 5)
//			getPoseFromEssentialMat(prevPointsL,curPointsL,R,t);
//		vector<Point2f> ppts,cpts;
//		for(int i=0;i<mask.rows;i++){
//			if(mask.at<int>(Point2f(i,0))){
//				ppts.push_back(prevPointsL[i]);
//				cpts.push_back(curPointsL[i]);
//			}
//		}
//		prevPointsL=ppts;
//		curPointsL=cpts;
//		cout << "#EprevPointsL="<<prevPointsL.size()<<endl;
//		cout << "#EcurPointsL="<<curPointsL.size()<<endl;
//		//essential pose estimation
//		cout << "R"<< R << endl;
//		cout << "t"<< t << endl;

		int cbad=0,nbad=0;
        prev3Dpts.clear();
        cur3Dpts.clear();
        vector<Point2f> cur2DforPnP,prev2Dpts;
        Point3f p3d,c3d;
        for(unsigned int i=0;i<prevPointsL.size();i++){
    		Point2f p2f1=prevPointsL[i];
    		Point2f c2f1= curPointsL[i];
    		Point2f ppx(p2f1.x,p2f1.y+cpImg.rows/2);//pixel of prev in cpImg
        	p3d=prevPointCloud.at<Point3f>(p2f1);
        	c3d= curPointCloud.at<Point3f>(c2f1);
        	Point3f dif3D=p3d-c3d;
        	float d=sqrt(dif3D.dot(dif3D));
        	float pz=-p3d.z;
        	float cz=-c3d.z;
        	if(pz!=-10000 && cz!=-10000 && d<2.5 && d>0.01){
        		int dis=prevDisp.at<unsigned short>(p2f1)>>4;//16-bit fixed-point disparity map (where each disparity value has 4 fractional bits)
        		Point2f p2f2=prevPointsL[i];
        		p2f2.x+=dis;
        		p2f2.y+=cpImg.rows/2;
        	    line(cpImg,ppx,p2f2,Scalar(0, 255, 0));
        	    line(cpImg,ppx,c2f1,Scalar(128, 128, 0));
        	    if(pz<190.5 && pz>0.0){
                	//cout << "dif3D="<<dif3D<<":"<<d<<endl;
        		    //putText(cpImg,to_string(pz)+":"+to_string(d),ppx,1,1,Scalar(0, 255, 255));
        		    //putText(cpImg,to_string(cz),c2f1,1,1,Scalar(255, 255, 0));
        		    prev3Dpts.push_back(p3d);
        		    cur3Dpts.push_back(c3d);
        		    prev2Dpts.push_back(p2f1);
        		    cur2DforPnP.push_back(c2f1);
            		circle(cpImg,ppx,2,Scalar(0, 255, 0));
            		circle(cpImg,c2f1,2,Scalar(255, 0, 0));
            	    line(cpImg,ppx,c2f1,Scalar(255,255,255));
        	    }
               	else{
                		circle(cpImg,ppx,3,Scalar(0, 128, 255));
                		nbad++;
                	}
        	}
        	else{
        		circle(cpImg,ppx,3,Scalar(0, 0, 255));
        		cbad++;
        	}
        }
        cout << prevPointsL.size() << endl;
        cout <<"#cbad disparity="<<cbad<<endl;
        cout <<"#nbad disparity="<<nbad<<endl;
        cout <<"prev3Dpts="<< prev3Dpts.size() << endl;
        cout <<"cur2DforPnP="<< cur2DforPnP.size() << endl;

		Mat rvec;
		vector<int> inliers;
		//Rodrigues(R,rvec);
		//tll=t;
		vector<Point2f> proj2DafterPnP;
		/*
		cv::projectPoints(prev3Dpts,rvec,tll,K,Mat(),proj2DafterPnP);
		for(unsigned int i=0;i<inliers.size();i++){
			int j=inliers[i];
			Point2f n2=cur2DforPnP[j]-proj2DafterPnP[j];
			float d2=n2.dot(n2);
			cout << cur2DforPnP[j] <<"="<<proj2DafterPnP[j]<<":"<<d2<<endl;
		}
		*/
		//bool okPnP=true;
		//okPnP=solvePnPRansac(prev3Dpts, cur2DforPnP, K, Mat(), rvec, tll,false,500,2,0.99,inliers, SOLVEPNP_EPNP/*SOLVEPNP_ITERATIVE*/ );
		//Rodrigues(rvec,Rll);
        //cout <<"inliers after fitting PnPRansac="<< inliers.size() << endl;
		//cout << "rvec"<< rvec << endl;
		//cout << "Rll"<< Rll << endl;
		//cout << "tll"<< tll << endl;
		//okPnP=solvePnPRansac(cur3Dpts, prev2Dpts, K, Mat(), rvec, tll,false,500,2,0.99,inliers, SOLVEPNP_EPNP/*SOLVEPNP_ITERATIVE*/ );
        //if(!okPnP)
		//    putText(dispC8,"******** NOOOOOOOOO **********",Point2f(400,180),1,2,Scalar(0, 0, 255));
		//reprojection error
		//cv::projectPoints(prev3Dpts,rvec,tll,K,Mat(),proj2DafterPnP);
		//for(unsigned int i=0;i<inliers.size();i++){
		//	int j=inliers[i];
		//	Point2f n2=cur2DforPnP[j]-proj2DafterPnP[j];
		//	float d2=n2.dot(n2);
		//	cout << cur2DforPnP[j] <<"="<<proj2DafterPnP[j]<<":"<<sqrt(d2)<<endl;
		//	circle(cpImg,cur2DforPnP[j],10,Scalar(0,255,255));
		//	circle(cpImg,proj2DafterPnP[j],10,Scalar(255,255,255));
		//}
    	Mat  cur3DMat=Mat( cur3Dpts).reshape(1);
    	Mat prev3DMat=Mat(prev3Dpts).reshape(1);
        rigidTransformation(cur3DMat,prev3DMat,Rll,tll);
        Rodrigues(Rll,rvec);
        rvec.at<float>(0.0)=0;
        rvec.at<float>(2.0)=0;
        Rodrigues(rvec,Rll);
		//Rodrigues(rvec,Rll);
        //cout <<"inliers after fitting PnPRansac="<< inliers.size() << endl;
		cout << "rvec"<< rvec.t() << endl;
		//cout << "Rll"<< Rll << endl;
		//cout << "tll"<< tll << endl;
		//cout << "tgl"<< tgl << endl;
		//cout << "Rgl"<< Rgl << endl;
		//update global transform
		Mat tll64;
		tll.at<float>(0,0)=0;
		tll.at<float>(1,0)=0;
		tll.convertTo(tll64,CV_64FC1);
		Mat dt=Rgl*tll64;
		//tgl = tgl + (Rgl*(scale*tll));
		tgl = tgl + dt;
		Rll.convertTo(Rll,CV_64FC1);
		Rgl = Rll*Rgl;
		//cout << "tgl"<< tgl << endl;
		//cout << "Rgl"<< Rgl << endl;
  		resize(cpImg, cpImg, Size(), 0.50,0.50);
        imshow("prevDisp",cpImg);
	}
	void step(Mat& pcurFrameL_c,Mat& pcurFrameR_c){
		curFrameL.copyTo(prevFrameL);		curFrameR.copyTo(prevFrameR);
		curFrameL_c.copyTo(prevFrameL_c);	curFrameR_c.copyTo(prevFrameR_c);
		pcurFrameL_c.copyTo(curFrameL_c);	pcurFrameR_c.copyTo(curFrameR_c);
		cvtColor(curFrameL_c, curFrameL, CV_BGR2GRAY);		cvtColor(curFrameR_c, curFrameR, CV_BGR2GRAY);
		prevKeypointsL = curKeypointsL;		prevKeypointsR = curKeypointsR;
		curDescriptorsL.copyTo(prevDescriptorsL);		curDescriptorsR.copyTo(prevDescriptorsR);
		prevPointsL = curPointsL;    		prevPointsR = curPointsR;

		detector->detect(curFrameL, curKeypointsL);
		detector->detect(curFrameR, curKeypointsR);
		extractor->compute(curFrameL, curKeypointsL, curDescriptorsL);
		extractor->compute(curFrameR, curKeypointsR, curDescriptorsR);

		findGoodMatches(prevKeypointsL,prevDescriptorsL,
				        curKeypointsL, curDescriptorsL,
						goodMatchesL,
						prevGoodKeypointsL,curGoodKeypointsL);
		findGoodMatches(prevKeypointsR,prevDescriptorsR,
				        curKeypointsR, curDescriptorsR,
						goodMatchesR,
						prevGoodKeypointsR,curGoodKeypointsR);

		prevPointsL.clear();
		curPointsL.clear();
		KeyPoint::convert(prevGoodKeypointsL, prevPointsL);
		KeyPoint::convert(curGoodKeypointsL,  curPointsL);
		prevPointsR.clear();
		curPointsR.clear();
		KeyPoint::convert(prevGoodKeypointsR, prevPointsR);
		KeyPoint::convert(curGoodKeypointsR,  curPointsR);


		if (prevPointsL.size() >5 && curPointsL.size() > 5 && prevPointsR.size() >5 && curPointsR.size() > 5){
			getPoseFromEssentialMat(prevPointsL,curPointsL,Rll,tll);
		    getPoseFromEssentialMat(prevPointsR,curPointsR,Rlr,tlr);
		}

        //Save actual global pose
		Rgl.copyTo(Rglprev);
		tgl.copyTo(tglprev);
		Rgr.copyTo(Rgrprev);
		tgr.copyTo(tgrprev);
		//Update global pose
		tgl = tgl + (Rll*(scale*tll));
		Rgl = Rll*Rgl;
		tgr = tgr + (Rlr*(scale*tlr));
		Rgr = Rlr*Rgr;

        //sm->compute(curFrameL, curFrameR, curDisp);
        //reprojectImageTo3D(curDisp, xyz, Q, true);
	}
	/*
	void stepF(Mat& pcurFrame_c){
		cur_frame.copyTo(prev_frame);
		cur_frame_c.copyTo(prev_frame_c);
		pcurFrame_c.copyTo(cur_frame_c);
		cvtColor(cur_frame_c, cur_frame, CV_BGR2GRAY);
		keypoints_1 = keypoints_2;
		descriptors_2.copyTo(descriptors_1);
		point1 = point2;

		detector->detect(cur_frame, keypoints_2);
		extractor->compute(cur_frame, keypoints_2, descriptors_2);
		matches.clear();
		good_matches.clear();

		try {
			matcher->knnMatch(descriptors_1, descriptors_2, matches, 2);
			ratioTest(matches, good_matches);
		} catch(Exception &e) {
			//cerr << "knnMatch error"<<endl;;
		}

		// TODO track features using Lucas Kanade
		// If no. of features falls below threshold
		// then recompute features and use knnMatch
		// to select good features
		// Repeat.


		// Retrieve 2D points from good_matches
		// Compute Essential Matrix, R & T
		good_keypoints_1.clear();
		good_keypoints_2.clear();
		point1.clear();
		point2.clear();
		for ( size_t m = 0; m < good_matches.size(); m++) {
			int i1 = good_matches[m].queryIdx;
			int i2 = good_matches[m].trainIdx;
			CV_Assert(i1 >= 0 && i1 < static_cast<int>(keypoints_1.size()));
            CV_Assert(i2 >= 0 && i2 < static_cast<int>(keypoints_2.size()));
            good_keypoints_1.push_back(keypoints_1[i1]);
            good_keypoints_2.push_back(keypoints_2[i2]);
		}
		KeyPoint::convert(good_keypoints_1, point1, vector<int>());
		KeyPoint::convert(good_keypoints_2, point2, vector<int>());


		if (point1.size() >5 && point2.size() > 5) {
			E = findEssentialMat(point2, point1, f, pp, RANSAC, 0.999, 1.0);
			recoverPose(E, point2, point1, R, t, f, pp);
			R_.copyTo(Rprev_);
			t_.copyTo(tprev_);
			t_ = t_ + (R_*(scale*t));
  			R_ = R*R_;
		}
	}
	void drawMatches(){
		cv::drawMatches( prev_frame_c, keypoints_1, cur_frame_c, keypoints_2,
               	good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  		resize(img_matches, img_matches, Size(), 0.25, 0.25);
	}*/

};

#endif /* DUALESTEREOVISUALODOMETRY_H_ */
