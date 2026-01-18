# Monte Carlo Simulations of the 3D Edwards–Anderson Spin Glass in Tubular Geometry

Check the values of the following macro(s) in the Makefile:
L: size of the lattice along the X and Y directions. Default value 16
LZ: size of the lattice along the Z direction. Default value 48
NBETAS: number of betas for the Parallel Tempering. Default value 40
OBC: Open Boundary Conditions flag.  Default value 1 
WEAK: type of OBC (see Supplementary Material in the paper). Default value: INDEPENDENT (see Supplementary Material in the paper)
STRONG: type of OBC (see Supplementary Material in the paper). Default value: INDEPENDENT (see Supplementary Material in the paper)
NO_CLUSTER: disable the Houdayer move (see Supplementary Material in the paper). Default value: 0 (the Houdayer move is active).

running make creates (with the default values) the executable: bin/EA_CUBE_L16_Lz48_NB40_NBPRE12_OBC

## License and citation

This project is licensed under the **MIT License**. See the `LICENSE` file for the full license text.

If you use this code, data, or derived results in a publication, please cite the associated article and/or this repository.  
Citation metadata is provided in the `CITATION.cff` file.

---

## Contact

Developed by M. Bernaschi, L.A. Fernandez, I. González-Adalid Pemartín, V. Martín-Mayor, G. Parisi, and F. Ricci-Tersenghi.

For questions, reach out at [isiglezadalid@gmail.com](mailto:isiglezadalid@gmail.com) or via GitHub issues.

