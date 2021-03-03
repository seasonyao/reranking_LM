If using `conda`, you can get this to work as follows:

```
conda create -n rerankLM python=3.8
conda activate rerankLM
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda install -c conda-forge jupyterlab
conda install -c conda-forge ipywidgets
conda install -c conda-forge matplotlib
conda install -c conda-forge jupyter_nbextensions_configurator
conda install pandas

git clone https://github.com/seasonyao/reranking_LM.git
cd reranking_LM

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```


