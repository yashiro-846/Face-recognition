#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

constexpr auto CV_AA = 16;

using namespace cv;
using namespace std;


int main(void)
{
	string filename = "haarcascades/haarcascade_frontalface_default.xml";
	string filename2 = "haarcascades/haarcascade_eye.xml";


	CascadeClassifier cascade, cascade2;
	cascade.load(filename); //正面顔情報が入っているカスケードファイル読み込み
	cascade2.load(filename2); //正面目情報が入っているカスケードファイル読み込み
		// 画像を格納するオブジェクトを宣言する
	cv::Mat frame, face, gauss, edge;
	Mat gray;



	// 動画ファイルを取り込むためのオブジェクトを宣言する
	cv::VideoCapture cap;
	cap.open("movie3.avi");

	// 動画ファイルが開けたか調べる
	if (cap.isOpened() == false) {
		printf("ファイルが開けません。\n");
		return -1;
	}



	for (;;) {
		// 1フレームを取り込む
		cap >> frame;				// cap から frame へ
		cap >> face;
		cap >> gauss;
		cap >> edge;


		// 画像から空のとき、無限ループを抜ける
		if (frame.empty() == true) {
			break;
		}




		vector<Rect> faces; //顔輪郭情報を格納場所
		vector<Rect> eye; //目輪郭情報を格納場所
		cascade.detectMultiScale(face, faces, 1.3, 3, 0, Size(20, 20)); //カスケードファイルに基づいて顔を検知する．検知した顔情報をベクトルfacesに格納

		for (int i = 0; i < faces.size(); i++) //検出した顔の個数"faces.size()"分ループを行う
		{
			rectangle(face, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 0, 255), 3, CV_AA); //検出した顔を赤色矩形で囲む
			cv::putText(face, "Face", Point(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));


			Rect gray(Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height));
			Mat roi = face(gray);

			cascade2.detectMultiScale(roi, eye, 1.01, 3, 0, Size(20, 20)); //カスケードファイルに基づいて顔を検知する．検知した顔情報をベクトルeyeに格納

			for (int i = 0; i < eye.size(); i++) //検出した目の個数"eye.size()"分ループを行う
			{
				rectangle(roi, Point(eye[i].x, eye[i].y), Point(eye[i].x + eye[i].width, eye[i].y + eye[i].height), Scalar(0, 255, 0), 3, CV_AA); //検出した顔を赤色矩形で囲む
				cv::putText(roi, "Eye", Point(eye[i].x + eye[i].width / 2, eye[i].y + eye[i].height + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
				//	circle(img, Point(eye[i].x + eye[i].width/2, eye[i].y + eye[i].height/2), radius(eye[i].width/2, eye[i].height/2), Scalar(0, 255, 0), 3, CV_AA); //検出した顔を赤色矩形で囲む

			}

		}




		// ガウシアンフィルタ
		// * Size(x, y)でx方向、y方向のフィルタサイズを指定する
		cv::GaussianBlur(frame, gauss, cv::Size(7, 7), 0.0);
		//エッジ抽出
		Canny(frame, edge, 50, 100);


		// ウィンドウに画像を表示する
		cv::imshow("顔検出", face);
		cv::imshow("ガウシアンフィルタ", gauss);
		cv::imshow("エッジ", edge);
		// 33ms待つ
		// * 引数にキー入力の待ち時間を指定できる。（ミリ秒単位）
		// * 引数が 0 または何も書かない場合、キー入力があるまで待ち続ける
		cv::waitKey(33);
	}

	return 0;
}





