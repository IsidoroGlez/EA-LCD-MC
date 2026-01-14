/******************************************************************************/
//
//  Adapted from GPU-based Swendsen-Wang multi-cluster algorithm for the simulation 
//  of classical spin systems
//
//  Copyright (C) 2015  Yukihiro Komura, Yutaka Okabe
//  RIKEN, Advanced Institute for Computational Science
//  Department of Physics, Tokyo Metropolitan University
//
//  This program is subject to the Standard CPC licence, 
//        (http://cpc.cs.qub.ac.uk/licence/licence.html)
//
//  Prerequisite: the NVIDIA CUDA Toolkit 5.0 or newer
//
//  Related publication:
//    Y. Komura and Y. Okabe, Comput. Phys. Commun. 183, 1155-1161 (2012).
//    doi: 10.1016/j.cpc.2012.01.017
//    Y. Komura,              Comput. Phys. Commun. 194, 54-58 (2015).
//    doi:10.1016/j.cpc.2015.04.015
//
//  We use the CUDA implementation for the cluster-labeling based on 
//  the workby Komura.
//    Y. Komura,             
//      Comput. Phys. Commun. 194, 54-58 (2015).
//
#include <string.h>
#include <curand.h>
#include "cudamacro.h"

#define DEVICONST __device__ __constant__

void ClusterInit(int, double, int, unsigned int, unsigned long long);
void ClusterStep(char **d_overlapAll[]);
void SWclose(void);
