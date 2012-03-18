#include "RectificationV120.h"

#include "LMOptimize.h"

int v120Rectify( const CvMat* _points1, const CvMat* _points2, CvSize imgSize, CvMat* _H1, CvMat* _H2 )
{
    CV_Assert(  CV_IS_MAT(_points1) && CV_IS_MAT(_points2) &&
                (_points1->rows == 1 || _points1->cols == 1) &&
                (_points2->rows == 1 || _points2->cols == 1) &&
                CV_ARE_SIZES_EQ(_points1, _points2) );
    // Points correspondences _points1, _points2
    // K1 K2 => need to know camera calibration matrices
    // assume they are the same and unit
    
    // Vector with angles and focal
    RectifyV120Functor functor(imgSize, _points1, _points2);
    CvVector *params = functor.allocParamVector(imgSize);
    CvVector *targetResult = functor.allocResultVector();

    // Use Levenberg Marquardt optimization
    levenbergMarquardt(functor, params, targetResult, 3000);

    // Unpack matrices 
    functor.setH1(_H1, params);
    functor.setH2(_H2, params);

    // Free memory
    cvReleaseMat(&params);
    cvReleaseMat(&targetResult);
}

// Constructor
RectifyV120Functor::RectifyV120Functor(const CvSize &imgSize, const CvMat* _points1, const CvMat* _points2)
: m_imgSize(imgSize)
, m_points1(_points1)
, m_points2(_points2)
, m_K1ori(NULL)
, m_K2ori(NULL)
{
    const double f1ori = sqrt(pow(m_imgSize.width,2) + pow(m_imgSize.height,2)); 
    const double f2ori = f1ori; // Same image size
    
    m_K1ori = cvCreateMat(3,3,CV_64F);
    m_K2ori = cvCreateMat(3,3,CV_64F);
    cvSetIdentity(m_K1ori);
    cvmSet(m_K1ori, 0, 0, f1ori);
    cvmSet(m_K1ori, 1, 1, f1ori);
    cvCopy(m_K1ori, m_K2ori); // Same camera matrices
}

// Destructor
RectifyV120Functor::~RectifyV120Functor()
{
    cvReleaseMat(&m_K1ori);
    cvReleaseMat(&m_K2ori);    
}

// Compute euler angles and return a matrix M = Rx*Ry*Rz
void RectifyV120Functor::eulerAngles(CvMat *M, double rx, double ry, double rz)
{
    double Rx[9];
    double Ry[9];
    double Rz[9];
    double tmp[9];

    CvMat _Rx = cvMat(3, 3, CV_64F, &Rx[0]);
    CvMat _Ry = cvMat(3, 3, CV_64F, &Ry[0]);
    CvMat _Rz = cvMat(3, 3, CV_64F, &Rz[0]);
    CvMat _tmp = cvMat(3, 3, CV_64F, &tmp[0]);
    
    cvSetIdentity(&_Rx);
    cvSetIdentity(&_Ry);
    cvSetIdentity(&_Rz);

    // Rx
    const double cx = cos(rx);
    const double sx = sin(rx);
    cvmSet(&_Rx, 1, 1, cx);
    cvmSet(&_Rx, 2, 2, cx);
    cvmSet(&_Rx, 1, 2, -sx);
    cvmSet(&_Rx, 2, 1, sx);

    // Ry
    const double cy = cos(ry);
    const double sy = sin(ry);
    cvmSet(&_Ry, 0, 0, cy);
    cvmSet(&_Ry, 2, 2, cy);
    cvmSet(&_Ry, 2, 0, -sy);
    cvmSet(&_Ry, 0, 2, sy);

    // z
    const double cz = cos(rz);
    const double sz = sin(rz);
    cvmSet(&_Rz, 0, 0, cz);
    cvmSet(&_Rz, 1, 1, cz);
    cvmSet(&_Rz, 0, 1, -sz);
    cvmSet(&_Rz, 1, 0, sz);
    
    // in the order Y-Z-X
    cvMatMul(&_Rz, &_Ry, &_tmp);
    cvMatMul(&_Rx, &_tmp, M);
}

// Returns a rectification matrix M from the other parameters
void RectifyV120Functor::rectificationMatrix(CvMat *M, CvMat *Kori, double rx, double ry, double rz, double f)
{
    // Rotation matrix
    CvMat *R = cvCreateMat(3,3, CV_64F);
    eulerAngles(R, rx, ry, rz);
   
    // Kori = Original projection matrix 
    // Knew = New projection matrix 
    CvMat *Knew = cvCreateMat(3,3,CV_64F);
    cvCopy(Kori, Knew);
    cvmSet(Knew,0,0, cvmGet(Kori,0,0)*pow(3.0,f));

    //Koriginal_inv = kn::inverseMatrixSVD(Koriginal[i]);
    CvMat *KoriInv = cvCreateMat(3,3,CV_64F);
    cvInv(Kori, KoriInv, CV_SVD);

    //H[i] = Knew[i] * R[i] * Koriginal_inv;
    CvMat *tmp = cvCreateMat(3,3,CV_64F);
    cvMatMul(R, KoriInv, tmp);
    cvMatMul(Knew,tmp,M); 
    
    // Free memory
    cvReleaseMat(&tmp);
    cvReleaseMat(&KoriInv);
    cvReleaseMat(&Knew);
    cvReleaseMat(&R);
}

// Function results=f(params)
void RectifyV120Functor::operator()(CvVector *params, CvVector *results)
{
    // Unpack matrices from params
    CvMat *_H1 = cvCreateMat(3,3,CV_64F); 
    CvMat *_H2 = cvCreateMat(3,3,CV_64F);
    setH1(_H1, params);
    setH2(_H2, params);

    // Compute points correspondance
    double error=0;
    double average=0;
    int nbMatches= std::max(m_points1->cols, m_points1->rows);
    CvMat *m1 = cvCreateMat( 1, nbMatches, CV_64FC2 );
    cvConvertPointsHomogeneous( m_points1, m1 );

    CvMat *m2 = cvCreateMat( 1, nbMatches, CV_64FC2 );
    cvConvertPointsHomogeneous( m_points2, m2 );
    const CvPoint2D64f* pt1 = (const CvPoint2D64f*)m1->data.ptr;
    const CvPoint2D64f* pt2 = (const CvPoint2D64f*)m2->data.ptr;
    const double* H1 = _H1->data.db;
    const double* H2 = _H2->data.db;
    double w1, x1, y1, w2, x2, y2;
    for(int i=0; i<nbMatches; i++)
    {
        // Compute y average
        // TODO test division/zero
        w1 = 1.0/(H1[6]*pt1[i].x + H1[7]*pt1[i].y + 1.);
        // not needed x1 = (H1[0]*pt1[i].x + H1[1]*pt1[i].y + H1[2])*w1;
        y1 = (H1[3]*pt1[i].x + H1[4]*pt1[i].y + H1[5])*w1;
        
        w2 = 1.0/(H2[6]*pt2[i].x + H2[7]*pt2[i].y + 1.);
        //not needed x2 = (H2[0]*pt2[i].x + H2[1]*pt2[i].y + H2[2])*w2;
        y2 = (H2[3]*pt2[i].x + H2[4]*pt2[i].y + H2[5])*w2;

        // Average y
        average = 0.5*(y1 + y2);
                
        // Error 
        error += pow(average-y1,2);
        error += pow(average-y2,2);
    }
   
    // Finally the error is :
    error /= static_cast<double>(nbMatches);
    cvSet(results, 0, error);

    // Free memory
    cvReleaseMat(&m1); 
    cvReleaseMat(&m2);
    cvReleaseMat(&_H1);
    cvReleaseMat(&_H2); 
}


