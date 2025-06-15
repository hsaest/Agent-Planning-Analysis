# Agent-Planning-Analysis

## Environments

```.sh
conda create -n agent-planning python==3.9.20
conda activate agent-planning
pip install -r requirements.txt
cd code/captum
pip install -e .

```

## Environments

Compute attribution scores.

```.sh
cd code/AttrScoreCalc
# for blocksworld
python blocksWorld.py 
# for travelplanner
python travelPlanner.py 
```

If you want to analyze the obtained attribution scores, just open and running two jupyter notebooks in code/analysis.

## Model Release

We fine-tune **Llama3.1-8B-Instruct** and **Qwen2-7B-Instruct** on BlocksWorld and TravelPlanner. The fine-tuned model weights are available on the HuggingFace 🤗.

- **[Llama-3.1-8B-Instruct-blocksworld-SFT](https://huggingface.co/hsaest/Llama-3.1-8B-Instruct-blocksworld-SFT)**
- **[Llama-3.1-8B-Instruct-travelplanner-SFT](https://huggingface.co/hsaest/Llama-3.1-8B-Instruct-travelplanner-SFT)**
- **[Qwen2-7B-Instruct-blocksworld-SFT](https://huggingface.co/hsaest/Qwen2-7B-Instruct-blocksworld-SFT)**
- **[Qwen2-7B-Instruct-travelplanner-SFT](https://huggingface.co/hsaest/Qwen2-7B-Instruct-travelplanner-SFT)**

## Citation Information

If our paper or related resources prove valuable to your research, we kindly ask for a citation. 

<a href="https://github.com/hsaest/Agent-Planning-Analysis"><img src="https://img.shields.io/github/stars/hsaest/Agent-Planning-Analysis?style=social&label=Agent-Planning-Analysis" alt="GitHub Stars"></a>

```
@article{xie2024revealing,
  title={Revealing the Barriers of Language Agents in Planning},
  author={Xie, Jian and Zhang, Kexun and Chen, Jiangjie and Yuan, Siyu and Zhang, Kai and Zhang, Yikai and Li, Lei and Xiao, Yanghua},
  journal={arXiv preprint arXiv:2410.12409},
  year={2024}
}
```
