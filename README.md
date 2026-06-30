# Figure Generation
- Scripts and required files to generate all main figures in ["Heterogeneous NF-κB activation and enhancer features shape transcription in Drosophila immunity"](https://www.cell.com/biophysj/fulltext/S0006-3495%2826%2900013-5).

- Links for cell TIFs + masks:
  - Figure 1: https://www.cellimagelibrary.org/groups/57531
  - Figure 3:
      - Dense Imaging (T1, T2, T3): https://www.cellimagelibrary.org/groups/57534
      - Sparse Imaging: https://www.cellimagelibrary.org/groups/57548  
- Links for files that exceed GitHub size limitation:
  - Fig 2 code's [goodcomp3_locations_dict_area.pkl file] to be saved (https://pages.github.com/](https://drive.google.com/file/d/1466sle5CYxscnlKvONHJSqz_ltZJ-9-2/view?usp=sharing) to be saved in file path: 'gitdir+'Figure 2 Files/Predictor Outputs 11.18/'

- Please make sure to update the line below to your save directory:
  - gitdir = 'G:/path/' 

- If figures are either (a) not plotting in your python interpretor, or (b) saving as blank PNGs, comment out the respective line within that figure section
  - (a): fig.savefig(savename, bbox_inches = 'tight', dpi=1000)
  - (b): plt.show() 
