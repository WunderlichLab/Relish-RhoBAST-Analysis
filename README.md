# Figure Generation
- Links for cell TIFs + masks:
  - Figure 1: https://www.cellimagelibrary.org/groups/57531
  - Figure 3:
      - Dense Imaging (T1, T2, T3): https://www.cellimagelibrary.org/groups/57534
      - Sparse Imaging: https://www.cellimagelibrary.org/groups/57548  

- Scripts and required files to generate all main figures in ["Heterogeneous NF-κB activation and enhancer features shape transcription in Drosophila immunity"](https://www.biorxiv.org/content/10.1101/2025.05.19.654881v1).

- Please make sure to update the line below to your save directory:
  - gitdir = 'G:/path/' 

- If figures are either (a) not plotting in your python interpretor, or (b) saving as blank PNGs, comment out the respective line within that figure section
  - (a): fig.savefig(savename, bbox_inches = 'tight', dpi=1000)
  - (b): plt.show() 
