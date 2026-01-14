#include "header.h"

void Parallel_Tempering(s_aleatorio_HQ_64bits * p, uint8_t * which_clon_this_beta,uint8_t * which_beta_this_clon,
			int *energy_PT, double *aceptancePT, double *attemptsPT, double *aceptancePTraw)
{
  uint8_t ibeta,iclon;
  uint8_t temp;
  int change;
  double pt,exppt; //Variables needed by true Parallel Tempering


  for(ibeta=0;ibeta<(NBETAS-1);ibeta++){

    attemptsPT[ibeta]++;
    
    pt=-(betas[ibeta+1]-betas[ibeta])*
      (energy_PT[which_clon_this_beta[ibeta+1]]-energy_PT[which_clon_this_beta[ibeta]]);
    

    _actualiza_aleatorio_HQ_escalar(p[0]);    
    change=(int)(p[0].final>>63);


    change=0;
    if (pt>=0){
      aceptancePT[ibeta]++;
      change=1;
    }else{
      exppt=exp(pt);      
      aceptancePT[ibeta]+=exppt;
      if((FNORM*p[0].final) < exppt )
	change=1;
    }

    if(change){ // change is acepted
      temp=which_clon_this_beta[ibeta];
      which_clon_this_beta[ibeta]=which_clon_this_beta[ibeta+1];
      which_clon_this_beta[ibeta+1]=temp;      
      aceptancePTraw[ibeta]++;
    }
  }

  //Actualizamos la permutacion inversa
  for (ibeta=0;ibeta<NBETAS;ibeta++){
    iclon=which_clon_this_beta[ibeta];    
    which_beta_this_clon[iclon]=ibeta; 
  }
}
