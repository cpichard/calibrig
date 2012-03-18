#ifndef __LMOPTIMIZE_H__
#define __LMOPTIMIZE_H__

#include "OCVUtils.h"

template<typename CalcFunctor>
void computeJacobian( CalcFunctor &function, CvVector *parameters, CvMat *J )
{
    CvVector *paramsTmp = cvCloneVector(parameters);
    CvVector *resultTmp = function.allocResultVector(); 
    // Compute function on current parameters
    CvVector *resultCurrent = function.allocResultVector(); 
    function(parameters, resultCurrent);
    
    double delta=0;
    for(int j=0; j<cvSize(parameters); ++j)
    {   
        // cf multiple view geometry, 2nd ed., page 602
        const double pj = std::fabs(cvGet(parameters, j)); 
        delta = std::min(pj*1.0e-4,1.0e-6); 
        delta = std::max(delta,1.0e-13);
       
        // Reset the tmp param values
        //aTmp = a;   
        cvCopy(parameters, paramsTmp); 
        
        // Augment only param j
        //aTmp[j] += delta;                     // P'[j] = P[j] + delta
        cvSet(paramsTmp, j, cvGet(paramsTmp, j)+delta);
        
        // Compute augmented result
        //kn::Vector<double> Xtmp = f(aTmp,b);  // f(P')
        function(paramsTmp, resultTmp); 
       
        // Compute gradient 
        //Xtmp = (resultTmp - resultCurrent) / delta;     // for every line i : dX_i/da_j
        // Set values 
        //J.setColumn(j,Xtmp);
        for(int i=0; i<cvSize(resultTmp);i++)
        {
            const double val = (cvGet(resultTmp, i)-cvGet(resultCurrent, i)) / delta; 
            cvmSet(J, i, j, val);
        }
    }

    cvReleaseVector(&resultCurrent);
    cvReleaseVector(&resultTmp);
    cvReleaseVector(&paramsTmp);
}

bool improvement(CvVector *eps, CvVector *newEps)
{
    double sumBefore = 0.0;
    double sumAfter  = 0.0;

    for(unsigned int i=0; i<cvSize(eps); ++i)
    {
        sumBefore += cvGet(eps,i) * cvGet(eps,i); 
        sumAfter  += cvGet(newEps,i) * cvGet(newEps,i); 
    }

    return sumBefore > sumAfter;
}

template<typename CalcFunctor>
int levenbergMarquardt( CalcFunctor &calcFunctor, CvVector *parameters, CvVector *targetResult, unsigned int maxIter)
{
    // Create a jacobian matrix of the parameters
    const int resultSize=cvSize(targetResult);
    const int paramSize=cvSize(parameters);
    CvMat *J = cvCreateMat(resultSize, paramSize, CV_64F);

    // Compute jacobian
    computeJacobian(calcFunctor, parameters, J);

    // Compute estimation of lambda
    CvMat *Jt = cvCreateTranspose(J);
    CvMat *JtJ = cvCreateMat(paramSize, paramSize, CV_64F);
    cvMatMul(Jt, J, JtJ);
    double average = 0;
    for(int i=0; i<resultSize; i++)
    {
        average += cvmGet(JtJ, i, i);
    }
    double lambda = 1.0e-3 * average/static_cast<double>(resultSize);

    // Start iterative process
    CvVector *epsilon = cvCreateVector(resultSize, CV_64F);
    CvVector *newEpsilon = cvCreateVector(resultSize, CV_64F);
    CvVector *deltaParameters = cvCreateVector(paramSize, CV_64F);
    CvVector *candidatesParameters = cvCreateVector(paramSize, CV_64F);
    CvVector *tmpResult = cvCreateVector(resultSize, CV_64F);
    CvMat *lambdaId = cvCreateMat(paramSize, paramSize, CV_64F);
    cvSetIdentity(lambdaId); 
    CvMat *augmented = cvCreateMat(paramSize, paramSize, CV_64F);
    CvVector *Jteps = cvCreateVector(paramSize, CV_64F);
    for(int iter=0; iter<maxIter; ++iter)
    {
        // Compute jacobian
        // Note: first time it's not needed, but whatever ...
        computeJacobian(calcFunctor, parameters, J);
        
        // Error vector         
        calcFunctor(parameters, tmpResult);
        cvSub(targetResult, tmpResult, epsilon); 
        
        // 
        int counter = 0;
        bool accepted = false;
        do
        {
            // Find delta
            cvTranspose(J, Jt);
            cvMatMul(Jt, J, JtJ);
            cvScaleAdd(lambdaId, cvScalar(lambda), JtJ, augmented);

            cvMatMul(Jt, epsilon, Jteps);
            cvSolve(augmented, Jteps, deltaParameters, CV_SVD);

            // Update parameters
            cvAdd(parameters, deltaParameters, candidatesParameters);

            // Compute new error vector
            calcFunctor(candidatesParameters, tmpResult);
            cvSub(targetResult, tmpResult, newEpsilon);

            // Check improvement
            if( improvement(epsilon, newEpsilon) == true )
            {
                lambda /= 10.0;
                cvCopy(candidatesParameters, parameters);
                accepted = true; // improve further
            }
            else
            {
                lambda *= 10.0;
                accepted=false;
            }

            // TODO 
            counter++;
            if(counter > 100) return 0;

        } while(accepted==false);
         
    }
    
    // Release memory
    cvReleaseVector(&Jteps);
    cvReleaseMat(&augmented);
    cvReleaseMat(&lambdaId);
    cvReleaseVector(&tmpResult);
    cvReleaseVector(&candidatesParameters);
    cvReleaseVector(&deltaParameters);
    cvReleaseVector(&newEpsilon);
    cvReleaseVector(&epsilon);
    cvReleaseMat(&JtJ);
    cvReleaseMat(&Jt);
    cvReleaseMat(&J); 

    return 0;
}


#endif//__LMOPTIMIZE_H__
