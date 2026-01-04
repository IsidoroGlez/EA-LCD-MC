#define MAIN
#include "../include/cudamacro.h"
#include "../include/header.h"

#include "../include/fill_bits.h"
#include "../Cluster/HoudayerXEA.cu"
//#define DEBUG
//#define GDB

cudaStream_t *stream;

int main(int argc, char **argv){

  if ( (argc > 1) && ( (strcmp(argv[1], "-h") == 0)||(strcmp(argv[1], "--help") == 0) ) ) {
    print_help(argv[0]);
  }

  // info sample and replica
  int sample, nbits, ir, iclon, ibeta, itm;
  
  // names of input files:
  char name_input[MAXSTR];
  char name_betas[MAXSTR];
  char name_lut[MAXSTR];
  char name_list[MAXSTR] = {'\0'};
  
  int ibin, it, imes, ihou;
  int iter;
  int ibit;
  
  uint32_t mi_jobID;
  int device,count_devices;
  unsigned long long max_time = 0;
  
  check_time(max_time);
  
  switch(argc){
  case 9:
    snprintf(name_list,MAXSTR,"%s",argv[8]);
  case 8:
    sscanf(argv[7],"%llu",&max_time);
  case 7:
    sscanf(argv[6],"%d",&device);
    snprintf(name_lut,MAXSTR,"%s",argv[5]);
    snprintf(name_input,MAXSTR,"%s",argv[4]);
    snprintf(name_betas,MAXSTR,"%s",argv[3]);
    sscanf(argv[2],"%d",&nbits);
    sscanf(argv[1],"%d",&sample);
    break;
  default:
    print_and_exit("Usage: %s isample nbits beta.dat input.in LUT device [max_time [list_samples]]\n",
		   argv[0]);
  }

  if((nbits<0) | (nbits>=MAXNSAMPLES))
    print_and_exit("Error: nbits out of range, 0 <= bit < %d\n", MAXNSAMPLES);
  
  if(sample<0)
    print_and_exit("Error: sample out of range, 0 <= sample\n");
  
  mi_jobID = (uint32_t) sample;
  if(mi_jobID>((1<<NUMBITS_PER_JOBID)-1))
    print_and_exit("Our Counter-based PRNG  cannot accomodate more than %d samples\n",
		   (1<<NUMBITS_PER_JOBID));
  
  MY_CUDA_CHECK(cudaGetDeviceCount( &count_devices) );
  if((device<0)||(device>=count_devices))
    print_and_exit("device=%d count_devices=%d\n",device,count_devices);
  
  MY_CUDA_CHECK( cudaSetDevice(device) );

  // Read INPUT DATA, BETAS and LUT 
  read_input(name_input);
  read_betas(name_betas);
  read_lut(name_lut, name_betas);
  read_list_samples(name_list, nbits);

  // Creating output path
  create_sample_path(sample, nbits);

  // Checking flag for backup and getting seeds
  check_and_prepare_simulation_backup(sample, nbits);

  printf("Betas and LUT have been read correctly.\n");
  printf("Initializating J's and Spins ...\n");
  fflush(stdout);
  
  handle_sigterm();
  create_running();

  //INIT EVERYTHING
  AllocHost(nbits);
  Init(nbits);

  //CHECK BACKUP OR NOT
  if(data.flag==0){
    backup(nbits);
  }
  
  if(write_seeds){
    write_seeds_in_file(nbits);
  }
  
  stop(&data.nbin); // Last chance for changing nbin

  //CUDA KERNELS & STREAMS
  if(NULL==(stream = (cudaStream_t*) malloc(nbits * sizeof(cudaStream_t))))
    print_and_exit("Problems allocating streams\n");
  
  for(ibit=0;ibit<nbits;ibit++){
    MY_CUDA_CHECK( cudaStreamCreate(&(stream[ibit])) );
  }
  // cudaFuncSetCacheConfig( /* ENERGY */ , cudaFuncCachePreferL1);
  // cudaFuncSetCacheConfig( /* MC */ , cudaFuncCachePreferL1);

  //COPY EVERITHING TO DEVICE MEMORY
  CopyConstOnDevice();
  CopyDataOnDevice(nbits);

  //MAIN LOOP
  send_PT_state_to_GPU(nbits);
#ifdef DEBUG
  Measure_Energy(nbits);

  int Ener, Ener_scalar;
  for(ibit=0;ibit<nbits;ibit++)
    for(ir=0;ir<NR;ir++)
      for(ibeta=0;ibeta<NBETAS;ibeta++){
	iclon = which_clon_this_beta[ibit][ir][ibeta];
	Ener = h_Ener[iclon+ir*NBETAS+ibit*NRNBETAS];
	Ener_scalar=calculate_scalar_energy(ibit,ir,iclon);
	if ( Ener != Ener_scalar ){
	  print_and_exit("DEBUG: Energy MSC and scalar differ in ibit=%d ir=%d ibeta=%d:escalar=%d, read=%d\n",
			 ibit, ir, ibeta, Ener_scalar, Ener);

	}
      }

#endif
  
  iter = 2*data.itcut*data.itmax*data.mesfr*data.ptfr;
  measure_time(nbits);
  
  for(ibin = data.itcut; ibin < data.nbin; ibin++){

    renew_running();
    
    for(ibit=0;ibit<nbits;ibit++)
      memset(average_Energy[ibit], 0, sizeof(double)*NRNBETAS);

    for(it = 0; it < data.itmax; it++){

      for(imes = 0; imes < data.mesfr; imes++){

	for(ihou = 0; ihou < data.houfr; ihou++){
	
	  //SYNCHRONIZE
	  MY_CUDA_CHECK( cudaDeviceSynchronize() );

	  iter = Monte_Carlo(mi_jobID, iter, nbits);

	  //SYNCHRONIZE
	  MY_CUDA_CHECK( cudaDeviceSynchronize() );

	  // KERNEL ENERGY
	  Measure_Energy(nbits);

	  //SYNCHRONIZE
	  MY_CUDA_CHECK( cudaDeviceSynchronize() );
	
	  // PT
	  for(ibit=0;ibit<nbits;ibit++)
	    for(ir=0;ir<NR;ir++)
	      Parallel_Tempering(&random_PT[ibit][ir],
				 which_clon_this_beta[ibit][ir],
				 which_beta_this_clon[ibit][ir],
				 h_Ener+(ir*NBETAS+ibit*NRNBETAS),
				 aceptancePT[ibit][ir],
				 attemptsPT[ibit][ir],
				 aceptancePTraw[ibit][ir]);

	  send_PT_state_to_GPU(nbits);
	}//ihou
#ifndef NO_HOUDAYER
	Houdayer(nbits); // WARNING: comment only for previous test
#endif	
      }//imes

#ifdef DEBUG
      for(ibit=0;ibit<nbits;ibit++){
	MY_CUDA_CHECK( cudaMemcpy(h_MSC_u_even[ibit], h_spin[ibit][0],
				  sizeof(MYWORD)*HALF_MSC_VNRNBETAS, cudaMemcpyDeviceToHost) );
	MY_CUDA_CHECK( cudaMemcpy(h_MSC_u_odd[ibit], h_spin[ibit][1],
				  sizeof(MYWORD)*HALF_MSC_VNRNBETAS, cudaMemcpyDeviceToHost) );

	unpacking_u(ibit);
	for(ir=0;ir<NR;ir++)
	  for(ibeta=0;ibeta<NBETAS;ibeta++){
	    iclon = which_clon_this_beta[ibit][ir][ibeta];
	    Ener = h_Ener[iclon+ir*NBETAS+ibit*NRNBETAS];
	    Ener_scalar=calculate_scalar_energy(ibit,ir,iclon);
	    if ( Ener != Ener_scalar ){
	      print_and_exit("Energy MSC and scalar differ in ibin=%d it=%d ibit=%d ir=%d ibeta=%d:escalar=%d, read=%d\n",
			     ibin, it, ibit, ir, ibeta, Ener_scalar, Ener);
	    }
	  }
      }
#endif
      
      // AVERAGE ENERGY
      Measure_Energy(nbits);
      
      for(ibit=0;ibit<nbits;ibit++){
	for(ir=0;ir<NR;ir++){
	  for(ibeta=0;ibeta<NBETAS;ibeta++){
	    iclon = which_clon_this_beta[ibit][ir][ibeta];
	    average_Energy[ibit][ibeta+ir*NBETAS] += (double) (h_Ener[iclon+ir*NBETAS+ibit*NRNBETAS]);
	  }
	}
      }

      // TRW
      if(it%frecTRW==(frecTRW-1)){
	itm = it/frecTRW;
	for(ibit=0;ibit<nbits;ibit++)
	  for(ir=0;ir<NR;ir++){
	    for(iclon=0;iclon<NBETAS;iclon++)
	      history_betas[ibit][ir][itm][iclon] = which_beta_this_clon[ibit][ir][iclon];
	  }
      }
      
    }//it

    // MEASURE PROGRAM SPEED
    measure_time(nbits);

    ignore_sigterm(); 

    // WRITE HISTORY
    write_history(ibin,nbits);	
    
    // WRITE ENERGY MEASURES
    for(ibit=0;ibit<nbits;ibit++)
      for(ir=0;ir<NR;ir++)
	for(ibeta=0;ibeta<NBETAS;ibeta++)
	  average_Energy[ibit][ibeta+ir*NBETAS] /= (double) data.itmax;
   
    
    write_measures(ibin, nbits);

    // WRITE CONF

    for(ibit=0;ibit<nbits;ibit++){
      MY_CUDA_CHECK( cudaMemcpy(h_MSC_u_even[ibit], h_spin[ibit][0],
				sizeof(MYWORD)*HALF_MSC_VNRNBETAS, cudaMemcpyDeviceToHost) );
      MY_CUDA_CHECK( cudaMemcpy(h_MSC_u_odd[ibit], h_spin[ibit][1],
				sizeof(MYWORD)*HALF_MSC_VNRNBETAS, cudaMemcpyDeviceToHost) );

    }
    
    write_conf(++data.itcut, nbits);

    stop(&data.nbin); // para o alarga el loop
    handle_sigterm();// ya estan todos los ficheros escritos

    // CHECK IF THERE IS ENOUGH TIME FOR OTHER ITERATION
    if(check_time(max_time))
      break;

  } //ibin
  
  // END SIMULATION
  ignore_sigterm(); 
  delete_running();
  
  printf("Simulation I=%d, nbits=%d finished at iteration %d\n",
	 sample, nbits,ibin);
  fflush(stdout);

  // FREE EVERYTHING
  FreeDevice(nbits);
  FreeHost(nbits);
  
  return 0;
}

int Monte_Carlo(uint32_t mi_jobID, int iter, int nbits)
{

  int imet, ir;
  uint64_t entropy, temporal, parity;

  dim3 dimGrid(numBlocks,nbits,1);
  dim3 dimBlock(numThreadsPerBlock,1,1);

  for(imet = 0; imet < data.ptfr; imet++){

    //KERNEL MC EVEN
    parity=0;
    _fill_bits;
    d_gen_rndbits_cuda<<<numBlocks,numThreadsPerBlock>>>(HALF_MSC_V,
							 NR,
							 NBETAS,
							 s_time_and_entropy,
							 seed_keys,
							 dev_lut_heat_bath,
							 rand_wht_h);


    d_oneMCstepBN_multisample<<<dimGrid, dimBlock>>>(d_spin,
						     parity,
						     d_J,
						     d_neig,
						     HALF_MSC_V,
						     NR,
						     NBETAS,
						     d_whichclone,
						     d_bianchi_rotate,
						     rand_wht_h);

    iter++;
    
    //KERNEL MC ODD
    parity=1;
    _fill_bits;
    d_gen_rndbits_cuda<<<numBlocks,numThreadsPerBlock>>>(HALF_MSC_V,
							 NR,
							 NBETAS,
							 s_time_and_entropy,
							 seed_keys,
							 dev_lut_heat_bath,
							 rand_blk_h);
    
    d_oneMCstepBN_multisample<<<dimGrid, dimBlock>>>(d_spin,
						     parity,
						     d_J,
						     d_neig,
						     HALF_MSC_V,
						     NR,
						     NBETAS,
						     d_whichclone,
						     d_neri_rotate,
						     rand_blk_h);
    
    iter++;
    
  }
  
  return iter;
  
}

void Houdayer(int nbits){

  static int InitCluster=true;
  
#if defined(TEST_HOUDAYER_ENERGY)
  static int *h_DBGEnerB=NULL, *h_DBGEnerA=NULL;
#endif
#if defined(TEST_HOUDAYER_CONF)
  static MYWORD hDBG_MSC_u_even[MAXNSAMPLES][HALF_MSC_VNRNBETAS], hDBG_MSC_u_odd[MAXNSAMPLES][HALF_MSC_VNRNBETAS];
#endif

#define BETATHRESHOLD 0.5
  if(InitCluster) {
     InitCluster=false;
     unsigned long long clusterseed=(unsigned long long) data.seed_Cluster;
     double betathreshold=BETATHRESHOLD;
     ClusterInit(NBETAS, betathreshold, nbits, numThreadsPerBlock, clusterseed);
#if defined(TEST_HOUDAYER_ENERGY)
     if(NULL==(h_DBGEnerB = (int *) malloc(sizeof(int)*NRNBETAS*nbits)))
        print_and_exit("Problems allocating h_DBGEnerB\n");
     if(NULL==(h_DBGEnerA = (int *) malloc(sizeof(int)*NRNBETAS*nbits)))
        print_and_exit("Problems allocating h_DBGEnerA\n");
#endif
  }

  unsigned int OverlapnumBlocks=(( (HALF_MSC_VNRNBETAS/NR)+numThreadsPerBlock-1 )/numThreadsPerBlock);  
  unsigned int UnpacknumBlocks=NBETAS*(( V+numThreadsPerBlock-1 )/numThreadsPerBlock);
  unsigned int PacknumBlocks=NBETAS*(( V+numThreadsPerBlock-1 )/numThreadsPerBlock);  

  dim3 OverlapdimGrid(OverlapnumBlocks,nbits,1);
  dim3 OverlapdimBlock(numThreadsPerBlock,1,1);

  dim3 UnpackdimGrid(UnpacknumBlocks,nbits,1);
  dim3 UnpackdimBlock(numThreadsPerBlock,1,1);

  dim3 PackdimGrid(PacknumBlocks,nbits,1);
  dim3 PackdimBlock(numThreadsPerBlock,1,1);  
  
  /* test Houdayer's move */
#if defined(TEST_HOUDAYER_ENERGY)
  MY_CUDA_CHECK( cudaDeviceSynchronize() );

  // KERNEL ENERGY
  Measure_Energy(nbits);
  
  //SYNCHRONIZE
  MY_CUDA_CHECK( cudaDeviceSynchronize() );
  memcpy(h_DBGEnerB,h_Ener,sizeof(int)*NRNBETAS*nbits);
#endif
#if defined(TEST_HOUDAYER_CONF)
   for(int ibit=0;ibit<nbits;ibit++){
     MY_CUDA_CHECK( cudaMemcpy(hDBG_MSC_u_even[ibit], h_spin[ibit][0],
			       sizeof(MYWORD)*HALF_MSC_VNRNBETAS, cudaMemcpyDeviceToHost) );
     MY_CUDA_CHECK( cudaMemcpy(hDBG_MSC_u_odd[ibit], h_spin[ibit][1],
			       sizeof(MYWORD)*HALF_MSC_VNRNBETAS, cudaMemcpyDeviceToHost) );
   }
#endif
   int r1, r2;
   for(int g=0; g<2; g++) { /* divide the replicas in two distinct groups */
     int ind1, tmp;
     int index[NR];
     for (int k = 0; k < NR/2; k++) { index[k]=k+(g*(NR/2)); }
     for (int k = 0; k < NR/2-1; k++) {
       ind1 = (k)+(random() % (NR/2-k));
       tmp=index[k];
        index[k]=index[ind1];
        index[ind1]=tmp;
     }
     r1=index[0]; /* pick 2 replicas in each group and computer the overlap */
     r2=index[1];
     d_computeoverlap<<<OverlapdimGrid,OverlapdimBlock>>>(d_spin,
	   			          d_overlap,
							  HALF_MSC_V,
							  NR,
							  NBETAS,
							  r1, /* temporary hack */
							  r2, /* temporary hack */
							  d_whichclonethisbeta);
     
     d_unpack<<<UnpackdimGrid, UnpackdimBlock>>>(d_overlap,
						 ds_uu,
						 HALF_MSC_V,
						 1,  /* a single overlap for the time being */
				  NBETAS);
     
     ClusterStep(ds_uu);				  
     
     d_pack<<<PackdimGrid, PackdimBlock>>>(d_overlap,
					   ds_uu,
					   HALF_MSC_V,
					   1,  /* a single overlap for the time being */
					   NBETAS);
     
     d_useoverlap<<<OverlapdimGrid,OverlapdimBlock>>>(d_spin,
						      d_overlap,
						      HALF_MSC_V,
						      NR,
						      NBETAS,
						      r1, /* temporary hack */
						      r2, /* temporary hack */
						      d_whichclonethisbeta);
   }
#if defined(TEST_HOUDAYER_CONF)
   d_computeoverlap<<<OverlapdimGrid,OverlapdimBlock>>>(d_spin,
							d_overlap,
							HALF_MSC_V,
							NR,
							NBETAS,
							r1, /* temporary hack */
							r2, /* temporary hack */
							d_whichclonethisbeta);
   
   d_unpack<<<UnpackdimGrid, UnpackdimBlock>>>(d_overlap,
					       ds_uu,
					       HALF_MSC_V,
					       1,  /* a single overlap for the time being */
					       NBETAS);
   
   ClusterStep(ds_uu);				  
   
   d_pack<<<PackdimGrid, PackdimBlock>>>(d_overlap,
					 ds_uu,
					 HALF_MSC_V,
					 1,  /* a single overlap for the time being */
					 NBETAS);
   
   d_useoverlap<<<OverlapdimGrid,OverlapdimBlock>>>(d_spin,
						    d_overlap,
						    HALF_MSC_V,
						    NR,
						    NBETAS,
						    r1, /* temporary hack */
						    r2, /* temporary hack */
						    d_whichclonethisbeta);
   
   for(int ibit=0;ibit<nbits;ibit++){
     MY_CUDA_CHECK( cudaMemcpy(h_MSC_u_even[ibit], h_spin[ibit][0],
			       sizeof(MYWORD)*HALF_MSC_VNRNBETAS, cudaMemcpyDeviceToHost) );
     MY_CUDA_CHECK( cudaMemcpy(h_MSC_u_odd[ibit], h_spin[ibit][1],
			       sizeof(MYWORD)*HALF_MSC_VNRNBETAS, cudaMemcpyDeviceToHost) );
     if(memcmp(h_MSC_u_even[ibit],hDBG_MSC_u_even[ibit],sizeof(MYWORD)*HALF_MSC_VNRNBETAS)!=0) {
       printf("conf changed for sample %d even\n",ibit);
       exit(1);
     }
     if(memcmp(h_MSC_u_odd[ibit],hDBG_MSC_u_odd[ibit],sizeof(MYWORD)*HALF_MSC_VNRNBETAS)!=0) {
       printf("conf changed for sample %d odd\n",ibit);
       exit(1);	   
     }
   }
#endif   
#if defined(TEST_HOUDAYER_ENERGY)
   MY_CUDA_CHECK( cudaDeviceSynchronize() );

   // KERNEL ENERGY
   Measure_Energy(nbits);
   
   //SYNCHRONIZE
   MY_CUDA_CHECK( cudaDeviceSynchronize() );
   memcpy(h_DBGEnerA,h_Ener,sizeof(int)*NRNBETAS*nbits);
   int ht=0;
   for(int ibit=0;ibit<nbits;ibit++) {
     for(int ibeta=0;ibeta<NBETAS;ibeta++){
       if((h_DBGEnerB[which_clon_this_beta[ibit][r1][ibeta]+0*NBETAS+ibit*NRNBETAS]+h_DBGEnerB[which_clon_this_beta[ibit][r2][ibeta]+1*NBETAS+ibit*NRNBETAS])!=
	  (h_DBGEnerA[which_clon_this_beta[ibit][r1][ibeta]+0*NBETAS+ibit*NRNBETAS]+h_DBGEnerA[which_clon_this_beta[ibit][r2][ibeta]+1*NBETAS+ibit*NRNBETAS])) {
	 printf("Sample %d, Beta %d differ\n",ibit,ibeta);
	 ht=1;
       }
     }
   }
   if(ht==1) {
     exit(1);
   } else {
     printf("Test Houdayer OK\n");
   }
#endif
   /*  end of test Houdayer's move */  

}

void Measure_Energy(int nbits)
{
  int ibit;
  
  MY_CUDA_CHECK( cudaMemset( d_Ener, 0, nbits*NRNBETAS*sizeof(int) ) );

  for(ibit=0;ibit<nbits;ibit++){
    
    d_computeEne<<<numBlocks,numThreadsPerBlock,numThreadsPerBlock*sizeof(MYWORD),stream[ibit]>>>(h_spin[ibit][0],
      h_spin[ibit][1],
      d_Ener+(ibit*NRNBETAS),
      d_sum, 0,
      d_J[ibit],
      d_neig[ibit],
      HALF_MSC_V,
      NR, NBETAS,
      d_bianchi_index,
      d_neri_index);
    
  }

  MY_CUDA_CHECK( cudaMemcpy(h_Ener, d_Ener, sizeof(int)*NRNBETAS*nbits,
			    cudaMemcpyDeviceToHost) );
}


// HOST MEMORY

void AllocHost(int nbits){

  int ibit, ir, ibeta;
  
  for(ibit=0; ibit<nbits;ibit++){

    if(NULL==(Jx[ibit] = (char *) malloc(sizeof(char)*V)))
      print_and_exit("Problems allocating Jx[%d]\n", ibit);

    if(NULL==(Jmx[ibit] = (char *) malloc(sizeof(char)*V)))
      print_and_exit("Problems allocating Jmx[%d]\n", ibit);
    
    if(NULL==(Jy[ibit] = (char *) malloc(sizeof(char)*V)))
      print_and_exit("Problems allocating Jy[%d]\n", ibit);

    if(NULL==(Jmy[ibit] = (char *) malloc(sizeof(char)*V)))
      print_and_exit("Problems allocating Jmy[%d]\n", ibit);
    
    if(NULL==(Jz[ibit] = (char *) malloc(sizeof(char)*V)))
      print_and_exit("Problems allocating Jz[%d]\n", ibit);

    if(NULL==(Jmz[ibit] = (char *) malloc(sizeof(char)*V)))
      print_and_exit("Problems allocating Jmz[%d]\n", ibit);
    
    if(NULL==(h_MSC_u_even[ibit] = (MYWORD *) malloc(sizeof(MYWORD)*HALF_MSC_VNRNBETAS)))
      print_and_exit("Problems allocating h_MSC_u_even[%d]\n", ibit);

    if(NULL==(h_MSC_u_odd[ibit] = (MYWORD *) malloc(sizeof(MYWORD)*HALF_MSC_VNRNBETAS)))
      print_and_exit("Problems allocating h_MSC_u_odd[%d]\n", ibit);

    if(NULL==(uu[ibit] = (char ***) malloc(sizeof(char **)*NR)))
      print_and_exit("Problems allocating uu[%d]\n", ibit);
    for(ir=0;ir<NR;ir++){
      if(NULL==(uu[ibit][ir] = (char **) malloc(sizeof(char *)*NBETAS)))
	print_and_exit("Problems allocating uu[%d][%d]\n", ibit, ir);
      for(ibeta=0;ibeta<NBETAS;ibeta++){
	if(NULL==(uu[ibit][ir][ibeta] = (char *) malloc(sizeof(char)*V)))
	  print_and_exit("Problems allocating uu[%d][%d][%d]\n", ibit, ir, ibeta);
      }//ibeta
    }//ir
  }//ibit
  
}

void FreeHost(int nbits){

    int ibit, ir, ibeta;

  for(ibit=0; ibit<nbits;ibit++){

    free(Jx[ibit]);
    free(Jmx[ibit]);
    free(Jy[ibit]);
    free(Jmy[ibit]);
    free(Jz[ibit]);
    free(Jmz[ibit]);
    
    free(h_MSC_u_even[ibit]);
    free(h_MSC_u_odd[ibit]);

    for(ir=0;ir>NR;ir++){
      for(ibeta=0;ibeta<NBETAS;ibeta++){
	free(uu[ibit][ir][ibeta]);
      }//ibeta
      
      free(uu[ibit][ir]);
    }//ir

    free(uu[ibit]);    
  }//ibit

}

// DEVICE MEMORY

void CopyConstOnDevice(void)
{

  size_t total;
  int scratch;
  
  MY_CUDA_CHECK( cudaMemcpyToSymbol(d_deltaE,&off[0],sizeof(int),0,cudaMemcpyHostToDevice) );
  MY_CUDA_CHECK( cudaMemcpyToSymbol(d_deltaN,&off[1],sizeof(int),0,cudaMemcpyHostToDevice) );	
  MY_CUDA_CHECK( cudaMemcpyToSymbol(d_deltaU,&off[2],sizeof(int),0,cudaMemcpyHostToDevice) );
  scratch=-off[0];
  MY_CUDA_CHECK( cudaMemcpyToSymbol(d_deltaO,&scratch,sizeof(int),0,cudaMemcpyHostToDevice) );
  scratch=-off[1];
  MY_CUDA_CHECK( cudaMemcpyToSymbol(d_deltaS,&scratch,sizeof(int),0,cudaMemcpyHostToDevice) );	
  scratch=-off[2];
  MY_CUDA_CHECK( cudaMemcpyToSymbol(d_deltaD,&scratch,sizeof(int),0,cudaMemcpyHostToDevice) );

  total = NBETAS*sizeof(s_lut_heat_bath);
  MY_CUDA_CHECK( cudaMalloc((void **) &dev_lut_heat_bath,total));
  MY_CUDA_CHECK( cudaMemcpy( dev_lut_heat_bath, h_LUT, total , cudaMemcpyHostToDevice) );  

  //  WARNING: Por qué no se copian las J's a una constante ?
  //  WARNING: Lo mismo para los index y las rotaciones blancas y negras...
  
}

void CopyDataOnDevice(int nbits){

  size_t  total;
  int ibit;
  
  Vicini *vicini;
  MYWORD jtemp[HALF_MSC_V];
  int neigtemp[HALF_MSC_V];
  int d, i, j;
  
  total = sizeof(unsigned int)*HALF_MSC_V;

  MY_CUDA_CHECK( cudaMalloc(&d_bianchi_index, total) );
  MY_CUDA_CHECK( cudaMalloc(&d_neri_index, total) );	
  MY_CUDA_CHECK( cudaMemcpy(d_bianchi_index, bianchi_index, total,cudaMemcpyHostToDevice) );
  MY_CUDA_CHECK( cudaMemcpy(d_neri_index, neri_index, total, cudaMemcpyHostToDevice) );

  total = sizeof(unsigned char)*HALF_MSC_V;
  
  MY_CUDA_CHECK( cudaMalloc(&d_bianchi_rotate, total) );
  MY_CUDA_CHECK( cudaMalloc(&d_neri_rotate, total) );
  MY_CUDA_CHECK( cudaMemcpy(d_bianchi_rotate, bianchi_rotate, total,cudaMemcpyHostToDevice) );
  MY_CUDA_CHECK( cudaMemcpy(d_neri_rotate, neri_rotate, total, cudaMemcpyHostToDevice) );

  for(ibit=0;ibit<nbits;ibit++){
    h_spin[ibit] = (MYWORD **)mmcuda((void ***)&d_spin[ibit],2,HALF_MSC_VNRNBETAS,sizeof(MYWORD),1);
    h_overlap[ibit] = (MYWORD **)mmcuda((void ***)&d_overlap[ibit],2,HALF_MSC_VNRNBETAS/NR,sizeof(MYWORD),1); /* for the time being a single overlap! */   
    h_uu[ibit] = (char **)mmcuda((void ***)&ds_uu[ibit],NBETAS,L*L*Lz,sizeof(char),1);    

    MY_CUDA_CHECK( cudaMemcpy(h_spin[ibit][0], h_MSC_u_even[ibit],
			      sizeof(MYWORD)*HALF_MSC_VNRNBETAS, cudaMemcpyHostToDevice) );
    MY_CUDA_CHECK( cudaMemcpy(h_spin[ibit][1], h_MSC_u_odd[ibit],
			      sizeof(MYWORD)*HALF_MSC_VNRNBETAS, cudaMemcpyHostToDevice) );
  
    h_J[ibit] = (MYWORD **)mmcuda((void ***)&d_J[ibit],2*DEGREE,HALF_MSC_V,sizeof(MYWORD),1);
    h_neig[ibit] = (int **)mmcuda((void ***)&d_neig[ibit],2*DEGREE,HALF_MSC_V,sizeof(int),1);
  
    for(d=0; d<2; d++) {
      vicini = (d==0)?viciniB:viciniN;
      for(i=0; i<DEGREE; i++) {
      for(j=0; j<HALF_MSC_V; j++) {
	jtemp[j]=vicini[j].J[ibit][i];
	neigtemp[j]=vicini[j].neig[i];
      }
      MY_CUDA_CHECK( cudaMemcpy(h_J[ibit][i+d*DEGREE], jtemp,
				sizeof(MYWORD)*HALF_MSC_V, cudaMemcpyHostToDevice) );    
      MY_CUDA_CHECK( cudaMemcpy(h_neig[ibit][i+d*DEGREE], neigtemp,
				sizeof(int)*HALF_MSC_V, cudaMemcpyHostToDevice) );
      }
    }
  
    MY_CUDA_CHECK( cudaMalloc(&d_whichclone[ibit],sizeof(unsigned int)*NRNBETAS) );
    MY_CUDA_CHECK( cudaMalloc(&d_whichclonethisbeta[ibit],sizeof(unsigned int)*NRNBETAS) );    
  }

  if(NULL==(h_Ener = (int *) malloc(sizeof(int)*NRNBETAS*nbits)))
    print_and_exit("Problems allocating h_Ener\n");
  MY_CUDA_CHECK( cudaMalloc(&d_Ener,sizeof(int)*NRNBETAS*nbits) );
  
  MY_CUDA_CHECK( cudaMalloc(&d_sum,sizeof(int)*MAX) );
  MY_CUDA_CHECK( cudaMemcpy(d_sum, sum, sizeof(sum), cudaMemcpyHostToDevice) );

  total = 3*HALF_MSC_VNRNBETAS*sizeof(uint64_t);
  MY_CUDA_CHECK(cudaMalloc(&rand_wht_h, total));
  MY_CUDA_CHECK(cudaMalloc(&rand_blk_h, total));
  
}

void send_PT_state_to_GPU(int nbits)
{

  unsigned int h_whichclone[NR*NBETAS],h_whichclonethisbeta[NR*NBETAS] ;
  int r, ib, ibit;

  for(ibit=0;ibit<nbits;ibit++){
    for(r=0;r<NR;r++)
      for(ib=0;ib<NBETAS;ib++) {
      	  h_whichclone[r+NR*ib] = (unsigned int) which_beta_this_clon[ibit][r][ib];
	  h_whichclonethisbeta[r+NR*ib] = (unsigned int) which_clon_this_beta[ibit][r][ib];	  
	}
  
    MY_CUDA_CHECK( cudaMemcpy(d_whichclone[ibit], h_whichclone,
			      sizeof(unsigned int)*NRNBETAS, cudaMemcpyHostToDevice) );
    MY_CUDA_CHECK( cudaMemcpy(d_whichclonethisbeta[ibit], h_whichclonethisbeta,
			      sizeof(unsigned int)*NRNBETAS, cudaMemcpyHostToDevice) );
			      
  }
}

void FreeDevice(int nbits){

  MY_CUDA_CHECK(cudaFree(d_bianchi_index));
  MY_CUDA_CHECK(cudaFree(d_neri_index));

  MY_CUDA_CHECK(cudaFree(d_bianchi_rotate));
  MY_CUDA_CHECK(cudaFree(d_neri_rotate));

  MY_CUDA_CHECK(cudaFree(d_sum));

  MY_CUDA_CHECK(cudaFree(dev_lut_heat_bath));

  MY_CUDA_CHECK(cudaFree(rand_wht_h));
  MY_CUDA_CHECK(cudaFree(rand_blk_h));

  MY_CUDA_CHECK(cudaFree(d_Ener));
  
  for(int ibit=0;ibit<nbits;ibit++){
    MY_CUDA_CHECK(cudaFree(d_whichclone[ibit]));
  }
}

#if defined(USE_LDG)
#define LDG(x) (__ldg(&(x)))
#else
#define LDG(x) (x)
#endif

__global__ void d_pack(MYWORD **overAll[],
	   		 char **ds_uu[],
			 int n,
			 int nover,
		  	 int nclone) {

  const unsigned int sampleId = blockIdx.y;

  MYWORD * __restrict__ evenOver = overAll[sampleId][0];
  MYWORD * __restrict__ oddOver =  overAll[sampleId][1];
  char ** __restrict__ d_uu=ds_uu[sampleId];

  const unsigned int tid = threadIdx.x+blockDim.x*blockIdx.x;

  const unsigned int iclone = tid/(n*nover);

  int tidModNNOver = tid - (iclone*n*nover);

  int whichover = tidModNNOver % nover;
  int whichsite = tidModNNOver / nover;

  if(iclone>=nclone || whichsite>=n) return;

  int x,y,z,resto,site;
  int bit, MSC_site;

  int alpha, beta, gamma, bx, by, bz;
  int aux;
  MSC_site=whichsite*2;
  unsigned int offset=(iclone*(nover)+whichover);
  gamma = MSC_site / MSC_S;
  resto = MSC_site - gamma*MSC_S;
  beta = resto / MSC_L;
  alpha = resto - beta*MSC_L;
  bit = 0;
  MYWORD tevenOver=0;
  MYWORD toddOver=0;  
  for(bz=0;bz<4;bz++){
      z = bz*MSC_Lz + gamma;
      for(by=0;by<4;by++){
        y = by*MSC_L + beta;
        for(bx=0;bx<4;bx++){
          aux = bx*MSC_L + alpha;
          x = aux | (0^((y^z)&1));
          site = x + y*L + z*S;
          tevenOver|=(((MYWORD)d_uu[offset][site])<<bit);

          x = aux | (1^((y^z)&1));
          site = x + y*L + z*S;
          toddOver|=(((MYWORD)d_uu[offset][site])<<bit);
          bit++;
        }//bx
      }//by
  }//bz
  evenOver[tid]=tevenOver;
  oddOver[tid]=toddOver;
}

__global__ void d_unpack(MYWORD **overlapAll[],
	   		 char **ds_uu[],
			 int n,
			 int nover,
		  	 int nclone) {

  const unsigned int sampleId = blockIdx.y;

  MYWORD * __restrict__ evenOver = overlapAll[sampleId][0];
  MYWORD * __restrict__ oddOver =  overlapAll[sampleId][1];
  char ** __restrict__ d_uu=ds_uu[sampleId];

  const unsigned int tid = threadIdx.x+blockDim.x*blockIdx.x;

  if(tid>=(V*NBETAS)) { return; }

  const unsigned int iclone = tid/V;

  const unsigned int cloneioff=iclone*n*nover;
  
  unsigned int ltid = tid - (iclone*V);
  int x,y,z;
  
  z=ltid/S;
  y=(ltid-(z*S))/Lx;
  x=ltid-(z*S)-(y*Lx);
  
  int alpha, beta, gamma, bx, by, bz;
  unsigned int parity;
  bz = z / MSC_Lz;
  gamma = z - bz*MSC_Lz;
  by = y / MSC_L;
  beta = y - by*MSC_L;
  bx = x / MSC_L;
  alpha = x - bx*MSC_L;
  parity=(x^y^z)&1;

  int bit, MSC_site;
  bit = bx + 4*by + 16*bz;
	      
  MSC_site = alpha + MSC_L*beta + gamma*MSC_S;
  MSC_site /= 2;

  unsigned int whichchar=bit/8;
  unsigned int shiftbit=bit%8;
  unsigned int offset;
  char *ptc;
  for(int ir=0; ir<nover; ir++) {
   ptc=(parity==0)?(char *)&(evenOver[cloneioff+(MSC_site*nover)+ir]):(char *)&(oddOver[cloneioff+(MSC_site*nover)+ir]);
   ptc+=whichchar;
   offset=(iclone*(nover)+ir);
   d_uu[offset][ltid]=(ptc[0]>>shiftbit)&1;
  }
}

__global__ void d_computeoverlap(MYWORD **spinAll[],
	   			 MYWORD **overlapAll[],
					  int n,
					  int nrep,
					  int nclone,
					  int r1,
					  int r2,
					  unsigned int **whichclone_thisbetaAll) {

  const unsigned int sampleId = blockIdx.y;

  MYWORD * __restrict__ evenSpin = spinAll[sampleId][0];
  MYWORD * __restrict__ oddSpin =  spinAll[sampleId][1];
  MYWORD * __restrict__ e_overlap = overlapAll[sampleId][0];
  MYWORD * __restrict__ o_overlap = overlapAll[sampleId][1];  

  const unsigned int *whichclone_thisbeta = whichclone_thisbetaAll[sampleId];

  const unsigned int tid = threadIdx.x+blockDim.x*blockIdx.x;

  int nover=1;      /* this maybe will be passed if multiple overlap need to be computed */
  int whichover=0;  /* this maybe will be passed if multiple overlap need to be computed */

  const unsigned int ibeta = tid/(n*nover);
  const unsigned int betaioff =  ibeta*n*nover;  

  int tidModNNOver = tid - (ibeta*n*nover);

  int whichsite = tidModNNOver / nover;
  if(ibeta>=nclone || whichsite>=n) return;
  
  unsigned int icloner1=whichclone_thisbeta[ibeta*nrep+r1];
  unsigned int icloner2=whichclone_thisbeta[ibeta*nrep+r2];
  
  const unsigned int clone1ioff = icloner1*n*nrep;
  const unsigned int clone2ioff = icloner2*n*nrep;  
  
  MYWORD spinr1 = LDG(evenSpin[clone1ioff + whichsite*nrep + r1]);
  MYWORD spinr2 = LDG(evenSpin[clone2ioff + whichsite*nrep + r2]);  

  e_overlap[betaioff + whichsite*nover + whichover] = spinr1 ^ spinr2;

  spinr1 = LDG(oddSpin[clone1ioff + whichsite*nrep + r1]);
  spinr2 = LDG(oddSpin[clone2ioff + whichsite*nrep + r2]);

  o_overlap[betaioff + whichsite*nover + whichover] = spinr1 ^ spinr2;

}

__global__ void d_useoverlap(MYWORD **spinAll[],
	   	             MYWORD **overlapAll[],
					  int n,
					  int nrep,
					  int nclone,
					  int r1,
					  int r2,
					  unsigned int **whichclone_thisbetaAll) {

  const unsigned int sampleId = blockIdx.y;

  MYWORD * __restrict__ evenSpin = spinAll[sampleId][0];
  MYWORD * __restrict__ oddSpin =  spinAll[sampleId][1];
  MYWORD * __restrict__ e_overlap = overlapAll[sampleId][0];
  MYWORD * __restrict__ o_overlap = overlapAll[sampleId][1];  

  const unsigned int *whichclone_thisbeta = whichclone_thisbetaAll[sampleId];

  const unsigned int tid = threadIdx.x+blockDim.x*blockIdx.x;

  int nover=1;      /* this maybe will be passed if multiple overlap need to be computed */
  int whichover=0;  /* this maybe will be passed if multiple overlap need to be computed */

  const unsigned int ibeta = tid/(n*nover);
  const unsigned int betaioff =  ibeta*n*nover;  

  int tidModNNOver = tid - (ibeta*n*nover);

  int whichsite = tidModNNOver / nover;
  if(ibeta>=nclone || whichsite>=n) return;
  
  unsigned int icloner1=whichclone_thisbeta[ibeta*nrep+r1];
  unsigned int icloner2=whichclone_thisbeta[ibeta*nrep+r2];
  
  const unsigned int clone1ioff = icloner1*n*nrep;
  const unsigned int clone2ioff = icloner2*n*nrep;  
  
  MYWORD spinr1 = LDG(evenSpin[clone1ioff + whichsite*nrep + r1]);
  MYWORD spinr2 = LDG(evenSpin[clone2ioff + whichsite*nrep + r2]);  

  spinr1^=e_overlap[betaioff + whichsite*nover + whichover];
  spinr2^=e_overlap[betaioff + whichsite*nover + whichover];
  evenSpin[clone1ioff + whichsite*nrep + r1]=spinr1;
  evenSpin[clone2ioff + whichsite*nrep + r2]=spinr2;

  spinr1 = LDG(oddSpin[clone1ioff + whichsite*nrep + r1]);
  spinr2 = LDG(oddSpin[clone2ioff + whichsite*nrep + r2]);

  spinr1^=o_overlap[betaioff + whichsite*nover + whichover];
  spinr2^=o_overlap[betaioff + whichsite*nover + whichover];
  oddSpin[clone1ioff + whichsite*nrep + r1]=spinr1;
  oddSpin[clone2ioff + whichsite*nrep + r2]=spinr2;
  
}


// KERNEL RNG
//Defines for philox_4x32_10 implementation (homemade)
#ifndef PHILOX_M4x32_0
#define PHILOX_M4x32_0 ((uint32_t)0xD2511F53)
#endif
#ifndef PHILOX_M4x32_1
#define PHILOX_M4x32_1 ((uint32_t)0xCD9E8D57)
#endif

#ifndef PHILOX_W32_0
#define PHILOX_W32_0 ((uint32_t)0x9E3779B9)
#endif
#ifndef PHILOX_W32_1
#define PHILOX_W32_1 ((uint32_t)0xBB67AE85)
#endif

#define _update_key {\
    key[0]+=PHILOX_W32_0; \
    key[1]+=PHILOX_W32_1;}

#define _update_state {\
  lo0=PHILOX_M4x32_0*v[0];                      \
  hi0=__umulhi(PHILOX_M4x32_0,v[0]);            \
  lo1=PHILOX_M4x32_1*v[2];                      \
  hi1=__umulhi(PHILOX_M4x32_1,v[2]);            \
  v[0]=hi1^v[1]^key[0];                         \
  v[1]=lo1;                                     \
  v[2]=hi0^v[3]^key[1];                         \
  v[3]=lo0;}

#define _philox_4x32_10 {\
  _update_state; \
  _update_key;   \
  _update_state; \
  _update_key;   \
  _update_state; \
  _update_key;   \
  _update_state; \
  _update_key;   \
  _update_state; \
  _update_key;   \
  _update_state; \
  _update_key;   \
  _update_state; \
  _update_key;   \
  _update_state; \
  _update_key;   \
  _update_state; \
  _update_key;   \
  _update_state;}

#define _obten_aleatorio_contador {					\
    uint32_t hi0,hi1,lo0,lo1;						\
    uint32_t key[2];							\
    key[0]=tid; key[1]=useed;	\
    v[0]=time_and_entropy.x;						\
    v[1]=time_and_entropy.y;						\
    v[2]=time_and_entropy.z;						\
    v[3]=time_and_entropy.w;						\
    _philox_4x32_10;							\
    time_and_entropy.x++; /* contador interno*/				\
    hi1=v[0]^v[2];							\
    lo1=v[1]^v[3];							\
    lsb[0]=((v[1]<<17)|(v[0]>>15))^hi1^((hi1<<21)|lo1>>11);		\
    lsb[1]=((v[0]<<17)|(v[1]>>15))^lo1^(lo1<<21);			\
    lsb[2]=(hi1<<28)|(lo1>>4);						\
    lsb[3]=(lo1<<28)|(hi1>>4);}


#define _bisection(Rmsb,Rlsb) {						\
    const uint32_t msb=Rmsb>>(32-NUMBITSPREBUSQUEDAS);			\
    uint32_t min,max,decision;						\
    decision=(uint32_t) (~(((Rmsb==dev_umbrales[254].x)?(Rlsb<dev_umbrales[254].y):(Rmsb<dev_umbrales[254].x))-1)); \
    min=dev_prebusqueda[msb]&255;					\
    max=dev_prebusqueda[msb]>>8;					\
    min=(decision&min)|((~decision)&255);				\
    max=(decision&max)|((~decision)&255);				\
    while((max-min)>1){							\
      npr=(max+min)>>1;							\
      decision=(uint32_t) (~(((Rmsb==dev_umbrales[npr].x)?(Rlsb<dev_umbrales[npr].y):(Rmsb<dev_umbrales[npr].x))-1)); \
      max=(decision&npr)|((~decision)&max);				\
      min=(decision&min)|((~decision)&npr);				\
    }									\
  decision=(uint32_t) (~(((Rmsb==dev_umbrales[min].x)?(Rlsb<dev_umbrales[min].y):(Rmsb<dev_umbrales[min].x))-1)); \
  npr=(decision&min)|(~decision&max);}

#define _get_rnd_word(bb) {				\
    uint32_t v[4],lsb[4];				\
    _obten_aleatorio_contador;				\
    const uint32_t rot=(lsb[0]+v[1]+lsb[2]+v[3])>>27;	\
    _bisection(v[0],lsb[0]);				\
    bb=npr;						\
    _bisection(v[1],lsb[1]);				\
    bb|=npr<<8;						\
    _bisection(v[2],lsb[2]);				\
    bb|=npr<<16;						\
    _bisection(v[3],lsb[3]);				\
    bb|=npr<<24;						\
    bb=(bb<<rot)|(bb>>(32-rot));}


__global__ void d_gen_rndbits_cuda(int n,
				   int nrep,
				   int nclone,
				   s_time s_time_and_entropy,
				   s_keys s_key,
				   s_lut_heat_bath *dev_lut_heat_bath,
				   uint64_t *rand_d) {

  const unsigned int tid = threadIdx.x+blockDim.x*blockIdx.x;
  const unsigned int tthreads = gridDim.x*blockDim.x;
  
  MYWORD scra;
  MYWORD b0,b1,b2;
  uint32_t b32t;

  int npr;

  unsigned int ibeta = tid / (n*nrep); //whichclone[iclone*nrep+whichrep];
  int ir=(tid-(ibeta*n*nrep))%nrep;
  
  //Select my counter and philox parameter
  uint4 time_and_entropy = s_time_and_entropy.vec[ir];
  uint32_t useed = s_key.my_key[ir];
  
  if (ibeta >= nclone) {
    return;
  }

  time_and_entropy.z += tid; 
  const uint2 *__restrict__ dev_umbrales = reinterpret_cast<const uint2 *>(dev_lut_heat_bath[ibeta].umbrales);
  const unsigned short *__restrict__ dev_prebusqueda = reinterpret_cast<const unsigned short *>(dev_lut_heat_bath[ibeta].prebusqueda);

  rand_d += tid;

  _get_rnd_word(b32t);
  scra=b32t;
  b0=scra<<32;
  _get_rnd_word(b32t);
  b0|=b32t;
  _get_rnd_word(b32t);
  scra=b32t;
  b1=scra<<32;
  _get_rnd_word(b32t);
  b1|=b32t;
  _get_rnd_word(b32t);
  scra=b32t;
  b2=scra<<32;
  _get_rnd_word(b32t);
  b2|=b32t;
  
  rand_d[0*tthreads] = b0;
  rand_d[1*tthreads] = b1;
  rand_d[2*tthreads] = b2;

  return;
}

// KERNEL ENERGY

#define MASK_E (0x1111111111111111ull)
#define MASK_N (0x000F000F000F000Full)
#define MASK_U (0x000000000000FFFFull)
#define MASK_O (0x8888888888888888ull)
#define MASK_S (0xF000F000F000F000ull)
#define MASK_D (0xFFFF000000000000ull)

#define NMASK_E (~MASK_E)
#define NMASK_N (~MASK_N)
#define NMASK_U (~MASK_U)
#define NMASK_O (~MASK_O)
#define NMASK_S (~MASK_S)
#define NMASK_D (~MASK_D)

__device__ MYWORD RotateE(MYWORD op) {
  return (op & MASK_E) << 3 | (op & NMASK_E) >> 1;
}

__device__ MYWORD RotateN(MYWORD op) {
  return (op & MASK_N) << 12 | (op & NMASK_N) >> 4;
}

__device__ MYWORD RotateU(MYWORD op) {
  return (op & MASK_U) << 48 | (op & NMASK_U) >> 16;
}

__device__ MYWORD RotateO(MYWORD op) {
  return (op & MASK_O) >> 3 | (op & NMASK_O) << 1;
}

__device__ MYWORD RotateS(MYWORD op) {
  return (op & MASK_S) >> 12 | (op & NMASK_S) << 4;
}

__device__ MYWORD RotateD(MYWORD op) {
  return (op & MASK_D) >> 48 | (op & NMASK_D) << 16;
}

#ifdef GDB
__device__ unsigned long long printBits(unsigned short num) {
    unsigned long long decimalValue = 0;
    unsigned long long factor = 1;  // Factor para construir el número en base 10

    for (int i = 15; i >= 0; i--) {  // Recorremos los 16 bits desde el más significativo
        decimalValue += ((num >> i) & 1) * factor;
        factor *= 10;  // Multiplicamos por 10 para construir el número en base 10
    }

    return decimalValue;
}
#endif

#define mask0 (NMASK_U)
#define maskLzm1 (NMASK_D)

__global__ void
__launch_bounds__(1024, 1)
  d_computeEne(MYWORD * __restrict__ newSpin,
	       MYWORD * __restrict__ oldSpin,
	       int * __restrict__ ene, int *sum,
	       const int dir,
	       MYWORD **J,
	       int **neig,
	       int n,
	       int nrep, int nclone,
	       unsigned int *p2p, unsigned int *p2np) {
  
  const unsigned int tid = threadIdx.x+blockDim.x*blockIdx.x;
  const unsigned int tthreads = gridDim.x*blockDim.x;
  const unsigned int iclone=tid/n/nrep;
  const unsigned int cloneioff=iclone*n*nrep;
  int whichrep=(tid-(iclone*n*nrep))%nrep;
  int whichsite=(tid-(iclone*n*nrep))/nrep;

  int whichgamma = (2*whichsite)/MSC_S;
  
#ifdef GDB
  int resto = (2*whichsite) - MSC_S*whichgamma;
  int whichbeta = resto / MSC_L;
  int whichalpha = resto - whichbeta*MSC_L;
#endif
  
  if(iclone>=nclone || whichsite>=n) return;

  int punto_n;
  int le=0;
  union Hack {unsigned long long lungo;  unsigned short corto[4];} hack;
#ifdef GDB
  Hack gdb_print;
#endif
  
  for(; whichsite<n; whichsite+=(tthreads/(nclone))) {
    const int punto_c=(whichsite)*nrep;
    MYWORD scra, scra2;

    //x+1
    punto_n=neig[DEGREE*dir+0][whichsite];
    scra=oldSpin[cloneioff+punto_n*nrep+whichrep];
    if((p2np[punto_n]-p2p[whichsite])!=d_deltaE){
      scra=RotateE(scra);
    }
    hack.lungo=newSpin[cloneioff+punto_c+whichrep]^(J[0+DEGREE*dir][whichsite]^scra);
    le+=(sum[hack.corto[0]]+sum[hack.corto[1]]+sum[hack.corto[2]]+sum[hack.corto[3]]);

    //x-1
    punto_n=neig[DEGREE*dir+3][whichsite];
    scra=oldSpin[cloneioff+punto_n*nrep+whichrep];
    if((p2np[punto_n]-p2p[whichsite])!=d_deltaO){
      scra=RotateO(scra);
    }
    hack.lungo=newSpin[cloneioff+punto_c+whichrep]^(J[3+DEGREE*dir][whichsite]^scra);
    le+=(sum[hack.corto[0]]+sum[hack.corto[1]]+sum[hack.corto[2]]+sum[hack.corto[3]]);
    
    //y+1
    punto_n=neig[DEGREE*dir+1][whichsite];
    scra=oldSpin[cloneioff+punto_n*nrep+whichrep];
    if((p2np[punto_n]-p2p[whichsite])!=d_deltaN)scra=RotateN(scra);
    hack.lungo=newSpin[cloneioff+punto_c+whichrep]^(J[1+DEGREE*dir][whichsite]^scra);
    le+=(sum[hack.corto[0]]+sum[hack.corto[1]]+sum[hack.corto[2]]+sum[hack.corto[3]]);

    if(dir==0) //Even site go to y+1
      scra2 = scra;

    //y-1
    punto_n=neig[DEGREE*dir+4][whichsite];
    scra=oldSpin[cloneioff+punto_n*nrep+whichrep];
    if((p2np[punto_n]-p2p[whichsite])!=d_deltaS)scra=RotateS(scra);
    hack.lungo=newSpin[cloneioff+punto_c+whichrep]^(J[4+DEGREE*dir][whichsite]^scra);
    le+=(sum[hack.corto[0]]+sum[hack.corto[1]]+sum[hack.corto[2]]+sum[hack.corto[3]]);
    
    if(dir==1) //Odd site go to y-1
      scra2 = scra;
    
    //z+1
    punto_n=neig[DEGREE*dir+2][whichsite];
    scra=oldSpin[cloneioff+punto_n*nrep+whichrep];
    if((p2np[punto_n]-p2p[whichsite])!=d_deltaU)scra=RotateU(scra);
    if(whichgamma==(MSC_Lz-1)){
      scra = (scra & maskLzm1) | (scra2 & (~maskLzm1));
    }
    hack.lungo=newSpin[cloneioff+punto_c+whichrep]^(J[2+DEGREE*dir][whichsite]^scra);
    le+=(sum[hack.corto[0]]+sum[hack.corto[1]]+sum[hack.corto[2]]+sum[hack.corto[3]]);
    
#ifdef GDB
    if((whichgamma==(MSC_Lz-1)) && (iclone==0)){
      gdb_print.lungo = newSpin[cloneioff+punto_c+whichrep];
      printf("S_i[%d, %d, %d]  = %016llu %016llu %016llu %016llu\n", whichalpha, whichbeta, whichgamma,
    	     printBits(gdb_print.corto[3]),
    	     printBits(gdb_print.corto[2]),
    	     printBits(gdb_print.corto[1]),
    	     printBits(gdb_print.corto[0]));
      gdb_print.lungo = scra;
      printf("S_j[%d, %d, %d]  = %016llu %016llu %016llu %016llu\n", whichalpha, whichbeta, whichgamma,
    	     printBits(gdb_print.corto[3]),
    	     printBits(gdb_print.corto[2]),
    	     printBits(gdb_print.corto[1]),
    	     printBits(gdb_print.corto[0]));
      
      gdb_print.lungo = J[2+DEGREE*dir][whichsite];
      printf("J_ij[%d, %d, %d] = %016llu %016llu %016llu %016llu\n", whichalpha, whichbeta, whichgamma,
    	     printBits(gdb_print.corto[3]),
    	     printBits(gdb_print.corto[2]),
    	     printBits(gdb_print.corto[1]),
    	     printBits(gdb_print.corto[0]));
      
      gdb_print.lungo = hack.lungo;
      printf("sJs[%d, %d, %d]  = %016llu %016llu %016llu %016llu\n", whichalpha, whichbeta, whichgamma,
    	     printBits(gdb_print.corto[3]),
    	     printBits(gdb_print.corto[2]),
    	     printBits(gdb_print.corto[1]),
    	     printBits(gdb_print.corto[0]));
    }
#endif
    
    //z-1
    punto_n=neig[DEGREE*dir+5][whichsite];
    scra = oldSpin[cloneioff+punto_n*nrep+whichrep];
    if((p2np[punto_n]-p2p[whichsite])!=d_deltaD)scra=RotateD(scra);
    if(whichgamma==0)
      scra = (scra & mask0) | (scra2 & (~mask0));
 
    hack.lungo=newSpin[cloneioff+punto_c+whichrep]^(J[5+DEGREE*dir][whichsite]^scra);
    le+=(sum[hack.corto[0]]+sum[hack.corto[1]]+sum[hack.corto[2]]+sum[hack.corto[3]]);
#ifdef GDB
    if((whichgamma==0) && (iclone==0)){
      gdb_print.lungo = newSpin[cloneioff+punto_c+whichrep];
      printf("S_i[%d, %d, %d]  = %016llu %016llu %016llu %016llu\n", whichalpha, whichbeta, whichgamma,
	     printBits(gdb_print.corto[3]),
	     printBits(gdb_print.corto[2]),
	     printBits(gdb_print.corto[1]),
	     printBits(gdb_print.corto[0]));
      gdb_print.lungo = scra2;
      printf("S_j[%d, %d, %d]  = %016llu %016llu %016llu %016llu\n", whichalpha, whichbeta, whichgamma,
	     printBits(gdb_print.corto[3]),
	     printBits(gdb_print.corto[2]),
	     printBits(gdb_print.corto[1]),
	     printBits(gdb_print.corto[0]));

      gdb_print.lungo = J[5+DEGREE*dir][whichsite];
      printf("J_ij[%d, %d, %d] = %016llu %016llu %016llu %016llu\n", whichalpha, whichbeta, whichgamma,
	     printBits(gdb_print.corto[3]),
	     printBits(gdb_print.corto[2]),
	     printBits(gdb_print.corto[1]),
	     printBits(gdb_print.corto[0]));
      
      gdb_print.lungo = hack.lungo;
      printf("sJs[%d, %d, %d]  = %016llu %016llu %016llu %016llu\n", whichalpha, whichbeta, whichgamma,
	     printBits(gdb_print.corto[3]),
	     printBits(gdb_print.corto[2]),
	     printBits(gdb_print.corto[1]),
	     printBits(gdb_print.corto[0]));
    }
#endif
    
  }
  int cubetti = 6*BITSINMYWORD - 2*le;
  atomicAdd(ene + (whichrep*nclone)+iclone, cubetti);    
}

// KERNEL FOR MC

__global__ void d_oneMCstepBN_multisample(MYWORD **spinAll[],
					  const int dir,
					  MYWORD **JAll[],
					  int **neigAll[],
					  int n,
					  int nrep,
					  int nclone,
					  unsigned int **whichcloneAll,
					  const unsigned char *rotate,
					  uint64_t *rand_h) {

  const unsigned int sampleId = blockIdx.y;

  MYWORD * __restrict__ newSpin = dir ? spinAll[sampleId][1] : spinAll[sampleId][0];
  MYWORD * __restrict__ oldSpin = dir ? spinAll[sampleId][0] : spinAll[sampleId][1];

  const MYWORD **J = (const MYWORD **) JAll[sampleId] + dir*DEGREE;
  const int **neig = (const int **) neigAll[sampleId];

  const unsigned int *whichclone = whichcloneAll[sampleId];

  // MYWORD * __restrict__ newSpin = dir ? spinAll[1] : spinAll[0];
  // MYWORD * __restrict__ oldSpin = dir ? spinAll[0] : spinAll[1];

  // const MYWORD **J = (const MYWORD **) (JAll + dir*DEGREE);
  // const int **neig = (const int **) neigAll;

  // const unsigned int *whichclone = whichcloneAll;

  const unsigned int tid = threadIdx.x+blockDim.x*blockIdx.x;
  const unsigned int tthreads = gridDim.x*blockDim.x;

  const unsigned int iclone = tid/(n*nrep);
  const unsigned int cloneioff = iclone*n*nrep;

  int tidModNNRep = tid - (iclone*n*nrep);

  int whichrep  = tidModNNRep % nrep;
  int whichsite = tidModNNRep / nrep;
  int whichgamma = (2*whichsite)/MSC_S;
  
  if(iclone>=nclone || whichsite>=n) return;

  unsigned int ibeta=whichclone[iclone*nrep+whichrep];

  rand_h += ibeta*(n*nrep) + tidModNNRep;

  neig += DEGREE*dir;

  oldSpin += cloneioff + whichrep;

  const MYWORD spintbu = LDG(newSpin[cloneioff + whichsite*nrep + whichrep]);

  const int punto_n0 = neig[0][whichsite];
  const int punto_n1 = neig[1][whichsite];
  const int punto_n2 = neig[2][whichsite];
  const int punto_n3 = neig[3][whichsite];
  const int punto_n4 = neig[4][whichsite];
  const int punto_n5 = neig[5][whichsite];

  MYWORD scra0 = oldSpin[punto_n0*nrep];
  MYWORD scra1 = oldSpin[punto_n1*nrep];
  MYWORD scra2 = oldSpin[punto_n2*nrep];
  MYWORD scra3 = oldSpin[punto_n3*nrep]; // negations moved into spintbu
  MYWORD scra4 = oldSpin[punto_n4*nrep];
  MYWORD scra5 = oldSpin[punto_n5*nrep];

  unsigned char rot = rotate[whichsite];
  if (rot & 0x01) scra0 = RotateE(scra0);
  if (rot & 0x02) scra1 = RotateN(scra1);
  if (rot & 0x04) scra2 = RotateU(scra2);
  if (rot & 0x08) scra3 = RotateO(scra3);
  if (rot & 0x10) scra4 = RotateS(scra4);
  if (rot & 0x20) scra5 = RotateD(scra5);

  MYWORD aux;
  if (dir==0)
    aux = scra1;
  else
    aux = scra4;

  if(whichgamma==(MSC_Lz-1)){
    scra2 = (scra2 & maskLzm1) | (aux & (~maskLzm1));
  }
  if(whichgamma==0)
    scra5 = (scra5 & mask0) | (aux & (~mask0));
  
  MYWORD f0 = spintbu ^ J[0][whichsite] ^ scra0; // negations moved into spintbu
  MYWORD f1 = spintbu ^ J[1][whichsite] ^ scra1;
  MYWORD f2 = spintbu ^ J[2][whichsite] ^ scra2;

  //Algebra del Metropolis
  MYWORD k1 = f0 ^ f1;
  MYWORD k2 = f0 & f1;
  MYWORD j1 = k1 ^ f2;
  MYWORD k3 = k1 & f2;
  MYWORD j2 = k2 ^ k3;
		
  f0 = spintbu ^ J[3][whichsite] ^ scra3;
  f1 = spintbu ^ J[4][whichsite] ^ scra4;
  f2 = spintbu ^ J[5][whichsite] ^ scra5;

  k1 = f0 ^ f1;
  k2 = f0 & f1;
  k3 = k1 & f2;
  MYWORD j3 = k1 ^ f2;
  MYWORD j4 = k2 ^ k3;

  MYWORD b0 = rand_h[0*tthreads];
  MYWORD b1 = rand_h[1*tthreads];
  MYWORD b2 = rand_h[2*tthreads];

  b1=b0&b1;
  b2=b1&b2;

  MYWORD id2=b1;
  MYWORD id1=(b0^b1)|b2;

  MYWORD j2ORj4=j2|j4;
  MYWORD flip=(j1&j3&id1) | ((j2ORj4|id2)&(j1|j3|id1))|((j2&j4)|(j2ORj4&id2)); 

  newSpin[cloneioff + whichsite*nrep + whichrep] = spintbu ^ flip;

}

#undef MASK_E
#undef MASK_N
#undef MASK_U
#undef MASK_O
#undef MASK_S
#undef MASK_D

#undef NMASK_E
#undef NMASK_N
#undef NMASK_U
#undef NMASK_O
#undef NMASK_S
#undef NMASK_D

// MMCUDA

#define MAKEMATR_RC 1
#if !defined(TRUE)
enum {FALSE, TRUE};
#endif
#if !defined(MAKEMATR_RC) 
#define MAKEMATR_RC 12
#endif

void **mmcuda(void ***rp, int r, int c, int s, int init) {
  int i;
  char **pc;
  short int **psi;
  int **pi;
  double **pd;
  char **d_pc;
  short int **d_psi;
  int **d_pi;
  double **d_pd;


  switch(s) {
  case sizeof(char):
    pc=(char **)malloc(r*sizeof(char *));
    if(!pc) create_error("error in makematr 1\n");
    MY_CUDA_CHECK( cudaMalloc( (void **) &d_pc, r*sizeof(char*) ) );
    for(i=0; i<r; i++) {
      MY_CUDA_CHECK( cudaMalloc( (void **) &pc[i], c*sizeof(char) ) );
      if(init) {
            MY_CUDA_CHECK( cudaMemset( pc[i], 0, c*sizeof(char) ) );
      }
    }
    MY_CUDA_CHECK( cudaMemcpy( d_pc, pc, r*sizeof(char *), cudaMemcpyHostToDevice ) );
    rp[0]=(void **)d_pc;
    return (void **)pc;
  case sizeof(short int):
    psi=(short int **)malloc(r*sizeof(short int*));
    if(!psi) create_error( "error in makematr 2\n");
    MY_CUDA_CHECK( cudaMalloc( (void **) &d_psi, r*sizeof(short int*) ) );
    for(i=0; i<r; i++) {
      MY_CUDA_CHECK( cudaMalloc( (void **) &psi[i], c*sizeof(short int) ) );
      if(init) {
            MY_CUDA_CHECK( cudaMemset( psi[i], 0, c*sizeof(short int) ) );
      }
    }
    MY_CUDA_CHECK( cudaMemcpy( d_psi, psi, r*sizeof(short int*), cudaMemcpyHostToDevice ) );
    rp[0]=(void **)d_psi;
    return (void **)psi;
  case sizeof(int):
    pi=(int **)malloc(r*sizeof(int*));
    if(!pi) create_error( "error in makematr 3\n");
    MY_CUDA_CHECK( cudaMalloc( (void **) &d_pi, r*sizeof(int*) ) );
    for(i=0; i<r; i++) {
      MY_CUDA_CHECK( cudaMalloc( (void **) &pi[i], c*sizeof(int) ) );
      if(init) {
            MY_CUDA_CHECK( cudaMemset( pi[i], 0, c*sizeof(int) ) );
      }
    }
    MY_CUDA_CHECK( cudaMemcpy( d_pi, pi, r*sizeof(int *), cudaMemcpyHostToDevice ) );
    rp[0]=(void **)d_pi;
    return (void **)pi;
  case sizeof(double):
    pd=(double **)malloc(r*sizeof(double*));
    if(!pd) create_error( "error in makematr 4 for %d rows\n",r);
    MY_CUDA_CHECK( cudaMalloc( (void **) &d_pd, r*sizeof(double*) ) );
    for(i=0; i<r; i++) {
      MY_CUDA_CHECK( cudaMalloc( (void **) &pd[i], c*sizeof(double) ) );
      if(init) {
            MY_CUDA_CHECK( cudaMemset( pd[i], 0, c*sizeof(double) ) );
      }
    }
    MY_CUDA_CHECK( cudaMemcpy( d_pd, pd, r*sizeof(double *), cudaMemcpyHostToDevice ) );
    rp[0]=(void **)d_pd;
    return (void **)pd;
  default:
    create_error("Unexpected size: %d\n",s);
    break;
  }
  return NULL;
}
