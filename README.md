# Agent-Planning-Analysis

The code and data will be fully released in two weeks. Stay tuned!



## Model Release

We fine-tune **Llama3.1-8B-Instruct** and **Qwen2-7B-Instruct** on BlocksWorld and TravelPlanner. The fine-tuned model weights are available on the HuggingFace 🤗.

- **[Llama-3.1-8B-Instruct-blocksworld-SFT](https://huggingface.co/hsaest/Llama-3.1-8B-Instruct-blocksworld-SFT)**
- **[Llama-3.1-8B-Instruct-travelplanner-SFT](https://huggingface.co/hsaest/Llama-3.1-8B-Instruct-travelplanner-SFT)**
- **[Qwen2-7B-Instruct-blocksworld-SFT](https://huggingface.co/hsaest/Qwen2-7B-Instruct-blocksworld-SFT)**
- **[Qwen2-7B-Instruct-travelplanner-SFT](https://huggingface.co/hsaest/Qwen2-7B-Instruct-travelplanner-SFT)**

|                    | Commonsense (Micro) | Commonsense (Macro) | Hard (Micro) | Hard (Macro) | Final Pass Rate |
|--------------------|:-------------------:|:-------------------:|:------------:|:------------:|:---------------:|
| **Direct Prompting**|                     |                     |              |              |                 |
| GPT-4o             |        84.7          |        31.1          |     53.6     |     31.1     |       7.8       |
| GPT-4o-Mini        |        84.4          |        22.2          |     42.4     |     20.0     |       2.2       |
| Llama3.1-8B        |        60.1          |         0.0          |      7.9     |      2.8     |       0.0       |
| Llama3.1-70B       |        82.8          |        18.9          |     33.1     |     16.1     |       2.2       |
| Qwen2-7B           |        49.9          |         1.1          |      2.1     |      0.0     |       0.0       |
| Qwen2-72B          |        74.8          |        11.7          |     23.8     |      8.9     |       1.7       |
| **Episodic Memory Updating** |            |                     |              |              |                 |
| GPT-4o             |        89.2          |        41.7          |     51.7     |     27.2     |       8.3       |
| GPT-4o-Mini        |        84.1          |        22.2          |     39.8     |     22.8     |       5.0       |
| Llama3.1-70B       |        84.9          |        23.9          |     39.5     |     24.4     |       6.1       |
| Qwen2-72B          |        75.6          |        13.8          |     28.8     |     10.6     |       3.3       |
| **Δ**              |       +1.8           |       +4.4           |    +1.7      |    +2.3      |      +2.2       |
| **Parametric Memory Updating** |           |                     |              |              |                 |
| GPT-4o             |        95.3          |        68.9          |     62.6     |     39.4     |      25.0       |
| GPT-4o-Mini        |        94.7          |        61.7          |     49.3     |     17.2     |      12.2       |
| Llama3.1-8B        |        78.3          |        17.8          |     19.3     |      6.1     |       3.8       |
| Qwen2-7B           |        59.0          |         0.6          |      0.2     |      0.0     |       0.0       |
| **Δ**              |      +12.1           |      +23.7           |    +6.4      |    +2.2      |      +7.8       |

## Citation Information

If our paper or related resources prove valuable to your research, we kindly ask for citation. 

<a href="https://github.com/hsaest/Agent-Planning-Analysis"><img src="https://img.shields.io/github/stars/hsaest/Agent-Planning-Analysis?style=social&label=Agent-Planning-Analysis" alt="GitHub Stars"></a>

```
@article{xie2024revealing,
  title={Revealing the Barriers of Language Agents in Planning},
  author={Xie, Jian and Zhang, Kexun and Chen, Jiangjie and Yuan, Siyu and Zhang, Kai and Zhang, Yikai and Li, Lei and Xiao, Yanghua},
  journal={arXiv preprint arXiv:2410.12409},
  year={2024}
}
```
