# Ling573 Deliverable 4

Multi-document summarization system implemented by team 1.

Team member: Haobo Gu, Haotian Zhu, Weifeng Jin, Yuanhe Tian



## Devtest

To run the system on **devtest** folder, just use the command below:

```
condor_submit src/D4.cmd
```

This command must be used at the root directory of the project.

The command will create a virtual environment and install all packages which are needed. You don't have to install packages by yourself. As a result, the first run may be quite slow. 

If you have all needed packages installed, every run will take about 2 hours. 

If you want to change input folder, please modify `src/D4.cmd`.

If you want to run the system on **evaltest** folder, you can also use:

```shell
condor_submit src/D4_eval.cmd
```



### Note

**The result of different runs may be slightly different**. This is because our PuLP would choose a random solution from all optimal solutions. 



## Outputs
The result of this run can be found in outputs folder.
All previous files with the same filename will be overwritten. 

We also include the output for a higher compression rate (0.8) in content selection part in `D4_0.8_eval` and `D4_0.8_dev` folder. Because the long running time of high compression rate, this output is not fully tested. 

## Documents
All needed documents can be found in doc folder, including reports, presentations and references.



## Results
The outputs are evaluated using ROUGE measurement. The evaluation result can be found in results folder. 

We also include the ROUGE result for a higher compression rate (0.8) in content selection part, in `D4_devtest_rouge_scores_0.8.out` and `D4_evaltest_rouge_scores_0.8.out`. Because the long running time of high compression rate, this result is not fully tested.