#include "../include/header.h"
#include <cassert>

char test_uu[NR][NBETAS][V];
MYWORD J[MSC_VDEGREE];
unsigned int MSC_neigh[MSC_VDEGREE];

void Init(int nbits){
  
  Init_Binario();
  
  Init_neighbours();
  
  Init_MSC_neighbours();
  Init_MSC_index_and_rotate();

  Init_Random(nbits);

  Init_u(nbits);
  packing_u(nbits);
  check_packing(nbits);

  Init_J(nbits);
  packing_J(nbits);

  calculate_blocks();

  Init_PT(nbits);
  
}

void generate_seed_vectors(void){
  uint64_t temp_seeds[MAXNSAMPLES];
  int i, ibit, ir;

  generate_seeds_from_one(&data.seed_J, temp_seeds, MAXNSAMPLES);  
  i = 0;
  for(ibit=0; ibit<MAXNSAMPLES; ibit++){
    if(ibit==list_samples[i]){
      seeds_J[i] = (randint) temp_seeds[ibit];
      i++;
    }
  }

  generate_seeds_from_one(&data.seed_u, temp_seeds, MAXNSAMPLES);  
  i = 0;
  for(ibit=0; ibit<MAXNSAMPLES; ibit++){
    if(ibit==list_samples[i]){
      seeds_u[i] = (randint) temp_seeds[ibit];
      i++;
    }
  }

  generate_seeds_from_one(&data.seed_MC, temp_seeds, NR);  
  for(ir=0; ir<NR; ir++){
    seeds_MC[ir] = (randint) temp_seeds[ir];
  }
  
}

void Init_Random(int nbits)
{
  int ir, ibit, seed_PT, i;
  uint64_t temp_seeds[MAXNSAMPLES];

  // Checking seeds
  for(ir=0;ir<NR;ir++){
    seeds_MC[ir] = comprueba_semilla(seeds_MC[ir]);
  }

  for(ibit=0;ibit<nbits;ibit++){
    seeds_J[ibit] = comprueba_semilla(seeds_J[ibit]);
    seeds_u[ibit] = comprueba_semilla(seeds_u[ibit]);
  }

  data.seed_Cluster = comprueba_semilla(data.seed_Cluster);

  // Init random generators for Metropolos and PT
  for(ir=0;ir<NR;ir++)
    Inicia_generadores_CPU(&random_PRC[ir],&random_xoshiro256pp[ir],
			   &seed_keys.my_key[ir],seeds_MC[ir]);

  for(ir=0;ir<NR;ir++){
    generate_seeds_from_one(&seeds_MC[ir], temp_seeds, MAXNSAMPLES);  

    i=0;
    for(ibit=0;ibit<MAXNSAMPLES;ibit++){
      if(ibit==list_samples[i]){
	seed_PT = (randint) temp_seeds[ibit];
	Init_Rand_HQ_64bits(&random_PT[i][ir],seed_PT);
	i++;
      }
    }
    if(i!=nbits)
      print_and_exit("Problems inizializating PT random generators\n");
  }  
}

void Init_neighbours(void)
{
  int j;

  for(j=0;j<Lx;j++){
    x_p[j]=1;
    x_m[j]=-1;

  }
  x_m[0]=Lx-1;
  x_p[Lx-1]=-x_m[0];

  for(j=0;j<Ly;j++){
    y_p[j]=Lx;
    y_m[j]=-Lx;
  }

  y_m[0]=(Ly-1)*Lx;
  y_p[L-1]=-y_m[0];


  for(j=0;j<Lz;j++){
    z_p[j]=Lx*Ly;
    z_m[j]=-Lx*Ly;
  }

  z_m[0]=(Lz-1)*Lx*Ly;
  z_p[Lz-1]=-z_m[0];
  
}

void Set_MSC_neigh(unsigned int vicini [], int punto_c,int punto_n, int dim){
  vicini[DEGREE*punto_c+dim]=punto_n;
  vicini[DEGREE*punto_n+dim+DIM]=punto_c;
  if(punto_c>=MSC_V) {
    fprintf(stderr,"Invalid punto_c (%d) in SetVicini",punto_c); exit(1);
  }
  if(punto_n>=MSC_V) {
    fprintf(stderr,"Invalid punto_n (%d) in SetVicini",punto_n); exit(1);
  }
}

int punto(int *ix, int *off){
  int dim,temp=ix[0];
  for (dim=1;dim<DIM;dim++)temp+=ix[dim]*off[dim];
  return temp;
}

void coordinate(int punto_c, int * ix,int * off){
  for (int dim=0;dim<DIM;dim++) { ix[dim]=((punto_c%off[dim+1])/off[dim]); } 
}

void sp(int iy[], int ix[] , int side[], int dir){
  for (int dim=0;dim<DIM;dim++) { iy[dim]=ix[dim]; }
  iy[dir]=(ix[dir]+1)%side[dir];
}

void Init_MSC_neighbours(void)
{
  int punto_c, punto_n, dim, volume;
  int ix[DIM];
  int iy[DIM];

  volume = 1;
  for (dim=0;dim<DIM-1;dim++){
    side[dim] = MSC_L;  
    off[dim] = volume;
    volume *= side[dim]; 
  }
  side[DIM-1] = MSC_Lz;  
  off[DIM-1] = volume;
  volume *= side[DIM-1]; 

  off[DIM] = volume;

  for(punto_c=0; punto_c<MSC_V;punto_c++){
    coordinate(punto_c,ix,off);
    for (dim=0;dim<DIM;dim++){
      sp(iy,ix,side,dim);
      punto_n=punto(iy,off);
      Set_MSC_neigh(MSC_neigh, punto_c, punto_n, dim);
    }
  }
  
}

void Init_MSC_index_and_rotate(void)
{

  int punto_c, punto_n, dim, volume, i, whichsite;
  int eoro[MSC_V];
  int skip, curbianchi, curneri;
  unsigned int low, high, medium, value;
  
  for(punto_c=0;punto_c<MSC_V;punto_c++){
    eoro[punto_c]=-1;
  }

  curbianchi = 0;
  curneri = 0;
  for(punto_c=0;punto_c<MSC_V;punto_c++){
    skip=0;
    for (dim=0;dim<DEGREE;dim++) {
      if(eoro[MSC_neigh[DEGREE*punto_c+dim]]!=-1) { skip=1; break; }
    }
    if(!skip) {
      if(curbianchi==HALF_MSC_V) {
	fprintf(stderr,"Unexpected number of bianchi\n");
	exit(1);
      }
      bianchi_index[curbianchi]=punto_c;
      curbianchi++;
      eoro[punto_c]=punto_c;
    }
  }
  for(i=0; i<MSC_V; i++) { if(eoro[i]==-1) {
      if(curneri==HALF_MSC_V) {
	fprintf(stderr,"Unexpected number of neri\n");
	exit(1);
      }
      neri_index[curneri]=i; curneri++;
    }
  }
  if(curbianchi!=curneri) {
    fprintf(stderr,"Unexpected value of bianchi (%d) and neri(%d)\n",curbianchi,curneri);
    exit(1);
  }

  for(punto_c=0; punto_c<HALF_MSC_V; punto_c++) {
    for (dim=0;dim<DEGREE;dim++) {
      value=MSC_neigh[DEGREE*bianchi_index[punto_c]+dim];
      low=0;
      high=curneri;
      while(low<high) {
	medium=(high+low)/2;
	if(neri_index[medium]<value) {
	  low=medium+1;
	} else { 
	  high=medium;
	}
      }
      if(neri_index[low]!=value) {
	fprintf(stderr,"Something wrong with bianco[%d]=%d, vicino %d not found\n",
		punto_c,bianchi_index[punto_c],value);
	exit(1);  
      }
      viciniB[punto_c].neig[dim]=low;
      value=MSC_neigh[DEGREE*neri_index[punto_c]+dim];
      low=0;
      high=curbianchi;
      while(low<high) {
	medium=(high+low)/2;
	if(bianchi_index[medium]<value) {
	  low=medium+1;
	} else { 
	  high=medium;
	}
      }
      if(bianchi_index[low]!=value) {
	fprintf(stderr,"Something wrong with nero[%d]=%d, vicino %d not found\n",
		punto_c,neri_index[punto_c],value);
	exit(1);
      }
      viciniN[punto_c].neig[dim]=low;
    }
  }

  for(whichsite=0; whichsite<HALF_MSC_V; whichsite++) {
    bianchi_rotate[whichsite]=0;
    punto_n=viciniB[whichsite].neig[0];
    if((neri_index[punto_n]-bianchi_index[whichsite])!=off[0]) bianchi_rotate[whichsite]=0x01;
    punto_n=viciniB[whichsite].neig[1];
    if((neri_index[punto_n]-bianchi_index[whichsite])!=off[1]) bianchi_rotate[whichsite]|=0x02;
    punto_n=viciniB[whichsite].neig[2];
    if((neri_index[punto_n]-bianchi_index[whichsite])!=off[2]) bianchi_rotate[whichsite]|=0x04;
    punto_n=viciniB[whichsite].neig[3];
    if((neri_index[punto_n]-bianchi_index[whichsite])!=-off[0]) bianchi_rotate[whichsite]|=0x08;
    punto_n=viciniB[whichsite].neig[4];
    if((neri_index[punto_n]-bianchi_index[whichsite])!=-off[1]) bianchi_rotate[whichsite]|=0x10;
    punto_n=viciniB[whichsite].neig[5];
    if((neri_index[punto_n]-bianchi_index[whichsite])!=-off[2]) bianchi_rotate[whichsite]|=0x20;
    neri_rotate[whichsite]=0;
    punto_n=viciniN[whichsite].neig[0];
    if((bianchi_index[punto_n]-neri_index[whichsite])!=off[0]) neri_rotate[whichsite]=0x01;
    punto_n=viciniN[whichsite].neig[1];
    if((bianchi_index[punto_n]-neri_index[whichsite])!=off[1]) neri_rotate[whichsite]|=0x02;
    punto_n=viciniN[whichsite].neig[2];
    if((bianchi_index[punto_n]-neri_index[whichsite])!=off[2]) neri_rotate[whichsite]|=0x04;
    punto_n=viciniN[whichsite].neig[3];
    if((bianchi_index[punto_n]-neri_index[whichsite])!=-off[0]) neri_rotate[whichsite]|=0x08;
    punto_n=viciniN[whichsite].neig[4];
    if((bianchi_index[punto_n]-neri_index[whichsite])!=-off[1]) neri_rotate[whichsite]|=0x10;
    punto_n=viciniN[whichsite].neig[5];
    if((bianchi_index[punto_n]-neri_index[whichsite])!=-off[2]) neri_rotate[whichsite]|=0x20;
  }
  
}

void Init_Binario(void){
    assert(sizeof(long long int)==4*sizeof(short));
    for (int cou=0;cou<MAX;cou++){
        int scra=cou;
        int ris=0;
        for(int i=0; i<16;i++){
            ris+=scra&1;
            scra>>=1;
        }
        sum[cou]=ris;
        //cout<<cou<<" "<<ris<<endl;
    }
};


void Init_u(int nbits)
{
  int site, iclon,irep, ibit;

  for(ibit=0;ibit<nbits;ibit++){
    // Inizializating random numbers
    Init_Rand_HQ_64bits(&random_u,seeds_u[ibit]);

    for(irep=0;irep<NR;irep++){
      for(iclon=0;iclon<NBETAS;iclon++)
	for(site=0;site<V;site++){
	  _actualiza_aleatorio_HQ_escalar(random_u);
	  if(random_u.final>>63){
	    uu[ibit][irep][iclon][site]=1;
	  }else{
	    uu[ibit][irep][iclon][site]=0;
	  }
	}
    }
  }
  
}

void packing_u(int nbits)
{
  int x,y,z,ir,iclon,tid,site,paridad;
  int alpha, beta, gamma, bx, by, bz, bit, ibit;
  int MSC_site;
  MYWORD * agujas[2];

  for(ibit=0;ibit<nbits;ibit++){
    memset(h_MSC_u_even[ibit],0,sizeof(MYWORD)*HALF_MSC_VNRNBETAS);
    memset(h_MSC_u_odd[ibit],0,sizeof(MYWORD)*HALF_MSC_VNRNBETAS);
    agujas[0] = h_MSC_u_even[ibit];
    agujas[1] = h_MSC_u_odd[ibit];

    for(ir=0;ir<NR;ir++){
      for(iclon=0;iclon<NBETAS;iclon++){
	site=0;
	for(z=0;z<Lz;z++){
	  bz = z / MSC_Lz;
	  gamma = z - bz*MSC_Lz;

	  for(y=0;y<Ly;y++){
	    by = y / MSC_L;
	    beta = y - by*MSC_L;

	    for(x=0;x<Lx;x++){
	      bx = x / MSC_L;
	      alpha = x - bx*MSC_L;
	      
	      paridad=(x^y^z)&1;
	      
	      bit = bx + 4*by + 16*bz;
	      
	      MSC_site = alpha + MSC_L*beta + gamma*MSC_S;
	      MSC_site /= 2;
	      
	      tid = ir + NR*MSC_site + HALF_MSC_VNR*iclon;
	      agujas[paridad][tid] |= ((MYWORD) uu[ibit][ir][iclon][site])<<bit;
	      site++;
	    }//x
	  }//y
	}//z
      }//iclon
    }//ir
  }//ibit
}

void unpacking_u(int ibit)
{
  int x,y,z,ir,iclon,tid,resto,site,paridad;
  int bit, MSC_site;

  int alpha, beta, gamma, bx, by, bz;
  int aux;
  MYWORD * agujas[2];

  agujas[0]=h_MSC_u_even[ibit];
  agujas[1]=h_MSC_u_odd[ibit];

  for(tid=0;tid<HALF_MSC_VNRNBETAS;tid++){
    iclon = tid/HALF_MSC_VNR;
    resto = tid-iclon*HALF_MSC_VNR;
    MSC_site = resto/NR;

    resto -= MSC_site*NR;
    ir=resto&(NR-1);

    MSC_site *= 2;

    gamma = MSC_site / MSC_S;
    resto = MSC_site - gamma*MSC_S;
    beta = resto / MSC_L;
    alpha = resto - beta*MSC_L;
    bit = 0;
    for(bz=0;bz<4;bz++){
      z = bz*MSC_Lz + gamma;
      for(by=0;by<4;by++){
        y = by*MSC_L + beta;
        for(bx=0;bx<4;bx++){
          aux = bx*MSC_L + alpha;
          paridad = 0;
          x = aux | (paridad^((y^z)&1));
          site = x + y*L + z*S;
          uu[ibit][ir][iclon][site]=(agujas[paridad][tid]>>bit)&1;

          paridad = 1;
          x = aux | (paridad^((y^z)&1));
          site = x + y*L + z*S;
          uu[ibit][ir][iclon][site]=(agujas[paridad][tid]>>bit)&1;

          bit++;
        }//bx
      }//by
    }//bz
  }//tid
}

void check_packing(int nbits)
{
  int ir,iclon,site,x,y,z, ibit;

  for(ibit=0;ibit<nbits;ibit++){
    for(ir=0;ir<NR;ir++)
      for(iclon=0;iclon<NBETAS;iclon++)
	memcpy(test_uu[ir][iclon], uu[ibit][ir][iclon], V*sizeof(char));
  
    unpacking_u(ibit);
    for(ir=0;ir<NR;ir++)
      for(iclon=0;iclon<NBETAS;iclon++){
	site=0;
	for(z=0;z<Lz;z++)
	  for(y=0;y<Ly;y++)
	    for(x=0;x<Lx;x++){
	      if(uu[ibit][ir][iclon][site]!=test_uu[ir][iclon][site]){
		printf("Wrong packing in ir=%d, iclon=%d\n",
		       ir,iclon);
		printf("site=%d, x=%d, y=%d,z=%d\n",site,x,y,z);
		print_and_exit("Must be: %d but: %d\n",uu[ir][iclon][site],
			       test_uu[ir][iclon][site]);
	      }
	      site++;
	    }
      }
  }
  printf("Packing spins well done.\n");
}

void Init_J(int nbits)
{
  int site, ibit;
  int countJ;

#ifdef MATTIS
  int x,y,z;
  int neigh_px,neigh_py,neigh_pz;
  printf("J Mattis\n");
#else
  printf("J random\n");
#endif
  
  for(ibit=0;ibit<nbits;ibit++){
    countJ = 0;
#ifdef MATTIS
    site=0;
    for(z=0;z<Lz;z++){
      neigh_pz=z_p[z];
      for(y=0;y<Ly;y++){
	neigh_py=y_p[y];
	for(x=0;x<Lx;x++){
	  neigh_px=x_p[x];
	  Jx[ibit][site]=uu[ibit][0][0][site]^uu[ibit][0][0][site+neigh_px];
	  Jy[ibit][site]=uu[ibit][0][0][site]^uu[ibit][0][0][site+neigh_py];
	  Jz[ibit][site]=uu[ibit][0][0][site]^uu[ibit][0][0][site+neigh_pz];	  
	  site++;
	}
      }
    }
#else

    Init_Rand_HQ_64bits(&random_J,seeds_J[ibit]);
    
    for(site=0;site<V;site++){
      _actualiza_aleatorio_HQ_escalar(random_J);
      if(random_J.final>>63){
	Jx[ibit][site]=1;
	countJ++;
      }else{
	Jx[ibit][site]=0;
      }
      _actualiza_aleatorio_HQ_escalar(random_J);
      if(random_J.final>>63){
	Jy[ibit][site]=1;
	countJ++;
      }else{
	Jy[ibit][site]=0;
      }
      _actualiza_aleatorio_HQ_escalar(random_J);
      if(random_J.final>>63){
	Jz[ibit][site]=1;
	countJ++;
      }else{
	Jz[ibit][site]=0;
      }  
    }
    
    if ( (data.flag==0) && (countJ != countJ0[ibit]) ){
      create_error("countJ and countJ0 differ (%d != %d --- ibit=%d)\n",
		   countJ, countJ0[ibit], list_samples[ibit]);
      print_and_exit("Problems generating J's\n");
    }

    if( (countJ0[ibit]!=0) && (countJ0[ibit]!=countJ) ){
      create_error("CountJ and countJ0 differ\n");
      print_and_exit("Problems generating J's\n");
    }
    
    countJ0[ibit] = countJ;
#endif
  }
}

void packing_J(int nbits)
{

  int x, y, z, site;
  int alpha, beta, gamma, bx, by, bz, bit;

  int pointer, punto_c, punto_n, dim;
  int MSC_site, ibit;

  for(ibit=0;ibit<nbits;ibit++){
  
    memset(J, 0, sizeof(MYWORD)*MSC_VDEGREE);
	
    site=0;
    for(z=0;z<Lz;z++){
      bz = z / MSC_Lz;
      gamma = z - bz*MSC_Lz;

      for(y=0;y<L;y++){
	by = y / MSC_L;
	beta = y - by*MSC_L;

	for(x=0;x<L;x++){
	  bx = x / MSC_L;
	  alpha = x - bx*MSC_L;

	  bit = bx + 4*by + 16*bz;

	  MSC_site = alpha + MSC_L*beta + gamma*MSC_S;

	  pointer = DEGREE*MSC_site;
	  punto_n = MSC_neigh[pointer];
	  J[pointer] |= ((Jx[ibit][site])&1ULL)<<bit;
	  J[DEGREE*punto_n+DIM] |= ((Jx[ibit][site])&1ULL)<<bit;

	  pointer++;
	  punto_n = MSC_neigh[pointer];
	  J[pointer] |= ((Jy[ibit][site])&1ULL)<<bit;
	  J[DEGREE*punto_n+1+DIM] |= ((Jy[ibit][site])&1ULL)<<bit;

	  pointer++;
	  punto_n = MSC_neigh[pointer];
	  J[pointer] |= ((Jz[ibit][site])&1ULL)<<bit;
	  J[DEGREE*punto_n+2+DIM] |= ((Jz[ibit][site])&1ULL)<<bit;
	  
	  site++;
	}
      }
    }
    
    for(punto_c=0; punto_c<HALF_MSC_V; punto_c++) {
      for (dim=0;dim<DEGREE;dim++) {
	viciniB[punto_c].J[ibit][dim]=J[DEGREE*bianchi_index[punto_c]+dim];
	viciniN[punto_c].J[ibit][dim]=J[DEGREE*neri_index[punto_c]+dim];	      
      }
    }
  }//ibit
}

void calculate_blocks(void)
{

  int error = 0;
  int i;
  numThreadsPerBlock = HALF_MSC_VNR;

  while(numThreadsPerBlock>CUDATHREADS || (numThreadsPerBlock%WARPSIZE)!=0) {
       numThreadsPerBlock/=2;
    }
    if(numThreadsPerBlock==0) {numThreadsPerBlock=WARPSIZE;}
    i=0;
    do {
	 numThreadsPerBlock>>=i;
   	 numBlocks = NBETAS*(( HALF_MSC_VNR+numThreadsPerBlock-1 )/numThreadsPerBlock);
	 i++;
    } while (numBlocks<1);
  
  
  if(error)
    print_and_exit("Problems with numThreadsPerBlock\n");
  
}

void Init_PT(int nbits)
{
  init_tempering(nbits);
  
  init_aceptances(nbits);
}

void init_tempering(int nbits)
{
  int ib,ic,ir, ibit;
  uint8_t * cual_beta_este_clon, * cual_clon_esta_beta;

  for(ibit=0;ibit<nbits;ibit++){
    for(ir=0;ir<NR;ir++){
    cual_beta_este_clon = which_beta_this_clon[ibit][ir];
    cual_clon_esta_beta = which_clon_this_beta[ibit][ir];
    
    for(ib=0;ib<NBETAS;ib++)
      cual_beta_este_clon[ib] = ib;
    
    for(ic=0;ic<NBETAS;ic++)
      cual_clon_esta_beta[ic] = ic;
    }

  }
}

void init_aceptances(int nbits)
{
  for(int ibit=0;ibit<nbits;ibit++){
    memset(aceptancePT[ibit], 0, sizeof(double)*NR*(NBETAS-1));
    memset(attemptsPT[ibit], 0, sizeof(double)*NR*(NBETAS-1));
    memset(aceptancePTraw[ibit], 0, sizeof(double)*NR*(NBETAS-1));
  }
}
