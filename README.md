# Ling573 Deliverable 3

Multi-document summarization system implemented by team 1.

Team member: Haobo Gu, Haotian Zhu, Weifeng Jin, Yuanhe Tian



## Devtest

To run the system on devtest folder, just use the command below:

```
condor_submit src/D3.cmd
```

This command must be used at the root directory of the project.

The command will create a virtual environment and install all packages which are needed. You don't have to install packages by yourself. As a result, the first run may be quite slow. 

If you have all needed packages installed, every run will take about 2 hours. 

If you want to change input folder, please modify `src/D3.cmd`.

### Note

**The result of different runs may be slightly different**. This is because our PuLP would choose a random solution from all optimal solutions. 



## Outputs
The result of this run can be found in outputs folder.
All previous files with the same filename will be overwritten. 



## Documents
All needed documents can be found in doc folder, including reports and references.



## Results
The outputs are evaluated using ROUGE measurement. The evaluation result can be found in results folder. 
