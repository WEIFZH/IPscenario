# Identifying IP Usage Scenarios: Problem, Data, and Benchmarks


This repo provides a dataset for IP Usage Scenarios prediction and codes of benchmarks as described in the paper:

> Identifying IP Usage Scenarios: Problem, Data, and Benchmarks
>
> [Weifeng Zhang](https://orcid.org/0000-0002-3109-0956), [Fan Zhou](https://dblp.org/pid/63/3122-2.html), [Kunpeng Zhang](http://www.terpconnect.umd.edu/~kpzhang/), [Zhiyuan Wang](https://orcid.org/0000-0001-7167-9055) and Yong Wang.
>
> Submitted for review  


## Dataset

we have compressed the datasets named as **dataset.zip** , you could refer to **documentation.xlsx** for more details. For running, you should unzip the file to "./data".


## Environmental Settings

Our experiments are conducted on Ubuntu 20.04, a single NVIDIA 1070Ti GPU, 32GB RAM, and Intel i7 8700K. 

```
torch = '1.3.1',
numpy = '1.19.1',
sklearn = '0.23.1',
pandas = '1.0.5'
```


## Usage

Here we take **Beijing** dataset as an example to demonstrate the usage.

### Preprocess

Before running benchmarks, you should convert the string data to numerical data:
```shell
python cate2num.py
```
then, you will get the beijing_cate2id.
### Run the benchmarks

For DT and SVM, you could run the IP_ML.py,
for D&CN and AutoInt, run the IP_DL.py,
and for NODE, please run with the command line:

```shell
cd node_scenario
python node_scenario.py --dataset "beijing"
# the dataset parameter choice is ["beijing", "shanghai", "sichuan", "illinois"]
```



## Cite

If you find our paper & code are useful for your research, please consider citing us:

```bibtex
@inproceedings{Zhang2021IPscenario, 
  author = {Weifeng Zhang, Fan zhou, Kunpeng Zhang, Zhiyuan Wang and Yong Wang}, 
  title = {Identifying IP Usage Scenarios: Problem, Data, and Benchmarks}, 
  booktitle = {Submitted for review},
  year = {2021},
}
```


## Acknowledgment

We would like to thank [DeepCTR](https://github.com/shenweichen/DeepCTR-Torch) for sharing their codes and [SHAP](https://github.com/slundberg/shap) for data analysing. 

## Contact

For any questions (as well as request for the pdf version) please open an issue or drop an email to: `weifzh At outlook Dot com`
