# Figure Generation
- Scripts to generate all main and supplementary figures and videos in ["Heterogeneous NF-κB activation and enhancer features shape transcription in Drosophila immunity"](https://www.biorxiv.org/content/10.1101/2025.05.19.654881v1).

- Please make sure to update the line below to your save directory:
  - gitdir = 'G:/path/' 

- If figures are either (a) not plotting in your python interpretor, or (b) saving as blank PNGs, comment out the respective line within that figure section
  - (a): fig.savefig(savename, bbox_inches = 'tight', dpi=1000)
  - (b): plt.show() 
