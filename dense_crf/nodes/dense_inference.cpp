/*
    Copyright (c) 2013, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
   THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>

#include "common.h"
#include "densecrf.h"
#include "ppm.h"
// Certainty that the groundtruth is correct
const float GT_PROB = 0.5;

// Simple classifier that is 50% certain that the annotation is correct
MatrixXf computeUnary(const VectorXi& lbl, int M) {
  const float u_energy = -log(1.0 / M);
  const float n_energy = -log((1.0 - GT_PROB) / (M - 1));
  const float p_energy = -log(GT_PROB);
  MatrixXf r(M, lbl.rows());
  r.fill(u_energy);
  // printf("%d %d %d \n",im[0],im[1],im[2]);
  for (int k = 0; k < lbl.rows(); k++) {
    // Set the energy
    if (lbl[k] >= 0) {
      r.col(k).fill(n_energy);
      r(lbl[k], k) = p_energy;
    }
  }

  std::cout << "unary size  " << r.rows() << " " << r.cols() << std::endl;
  // std::cout<<"unary row sum  "<<r.rowwise().sum()<<std::endl;

  return r;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    printf("Usage: %s image annotations output\n", argv[0]);
    return 1;
  }
  // Number of labels
  const int M =
      5;  // TODO. determine it based on your actual situations. lower, faster
  // Load the color image and some crude annotations (which are used in a simple
  // classifier)
  int W, H, GW, GH;

  unsigned char* im = readPPM(argv[1], W, H);  // input raw rgb images
  if (!im) {
    printf("Failed to load image!\n");
    return 1;
  }
  unsigned char* anno =
      readPPM(argv[2], GW, GH);  // input annotation images, also rgb
  if (!anno) {
    printf("Failed to load annotations!\n");
    return 1;
  }
  if (W != GW || H != GH) {
    printf("Annotation size doesn't match image!\n");
    return 1;
  }

  VectorXi templabel = getLabeling(anno, W * H, M);
  std::cout << "max label " << int(templabel.maxCoeff()) << std::endl;

  /////////// Put your own unary classifier here! ///////////
  MatrixXf unary = computeUnary(templabel, M);
  ///////////////////////////////////////////////////////////
  std::cout << " marker three" << std::endl;

  std::clock_t start;
  double duration;
  start = std::clock();

  std::cout << "lalalalalala" << std::endl;

  // Setup the CRF model
  std::cout << W << " " << H << "  " << M << std::endl;

  DenseCRF2D crf(W, H, M);
  // Specify the unary potential as an array of size W*H*(#classes)
  // packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
  crf.setUnaryEnergy(unary);
  // add a color independent term (feature = pixel location 0..W-1, 0..H-1)
  // x_stddev = 3
  // y_stddev = 3
  // weight = 3
  crf.addPairwiseGaussian(
      2, 2,
      new PottsCompatibility(
          8));  // crf.addPairwiseGaussian( 3, 3, new PottsCompatibility( 3 ) );
  // add a color dependent term (feature = xyrgb)
  // x_stddev = 60
  // y_stddev = 60
  // r_stddev = g_stddev = b_stddev = 20
  // weight = 10
  crf.addPairwiseBilateral(
      20, 20, 20, 20, 20, im,
      new PottsCompatibility(10));  // crf.addPairwiseBilateral( 40, 40, 13, 13,
                                    // 13, im, new PottsCompatibility( 10 ) );

  // Do map inference
  // 	MatrixXf Q = crf.startInference(), t1, t2;
  // 	printf("kl = %f\n", crf.klDivergence(Q) );
  // 	for( int it=0; it<5; it++ ) {
  // 		crf.stepInference( Q, t1, t2 );
  // 		printf("kl = %f\n", crf.klDivergence(Q) );
  // 	}
  // 	VectorXi map = crf.currentMap(Q);
  VectorXi map = crf.map(10);
  // Store the result
  unsigned char* res = colorize(map, W, H);

  // TODO change res to Eigen matrix

  std::cout << "map max element is " << map.maxCoeff() << std::endl;

  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  std::cout << "takes time: " << duration << '\n';

  writePPM(argv[3], W, H, res);

  delete[] im;
  delete[] anno;
  delete[] res;
}
